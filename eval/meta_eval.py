from __future__ import print_function

import numpy as np
import scipy
from scipy.stats import t
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from .util import FewShotBatch


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out

def meta_test_finetune(net, testloader, use_logit=True, is_norm=True, classifier='LR', fewshot_batch_size = 64):
    acc = []
    losses = []
    optimizer = optim.Adam(net.parameters())
    criterion = nn.BCEWithLogitsLoss()
    
    
    for idx, data in enumerate(tqdm(testloader, ascii = True)):
        net.train()
        support_xs, support_ys, query_xs, query_ys = data
        support_xs = support_xs.cuda()
        query_xs = query_xs.cuda()
        batch_size, _, height, width, channel = support_xs.size()
        support_xs = support_xs.view(-1, height, width, channel)
        query_xs = query_xs.view(-1, height, width, channel)

        if use_logit:
            query_features = net(query_xs).view(query_xs.size(0), -1)
        else:
            feat_query, _ = net(query_xs, is_feat=True)
            query_features = feat_query[-1].view(query_xs.size(0), -1)

        if is_norm:
            query_features = normalize(query_features)


        sup_xs_loader = DataLoader(FewShotBatch(support_xs), batch_size = fewshot_batch_size)

        support_features = []

        for idx_sup, sup in enumerate(sup_xs_loader):
            if use_logit:
                support_feature = net(sup).view(sup.size(0), -1)
            else:
                feat_support, _ = net(sup, is_feat=True)
                support_feature = feat_support[-1].view(sup.size(0), -1)

            if is_norm:
                support_feature = normalize(support_feature)
            support_features.append(support_feature)

        support_features = torch.cat(support_features, dim = 0)
        support_features = support_features.detach().cpu().numpy()
        query_features = query_features.detach().cpu().numpy()

        support_ys = support_ys.view(-1).numpy()
        query_ys = query_ys.view(-1).numpy()

        if classifier == 'LR':
            clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000,
                                        multi_class='multinomial')
            clf.fit(support_features, support_ys)

            # Fine tuning with support
            optimizer.zero_grad()
            support_ys_pred = clf.predict(support_features)
            loss = -criterion(torch.tensor(support_ys_pred, requires_grad= True, dtype=torch.float64), 
                             torch.tensor(support_ys, requires_grad=True, dtype=torch.float64)
                            )
            loss.backward()
            optimizer.step()
            query_ys_pred = clf.predict(query_features)
        elif classifier == 'NN':
            optimizer.zero_grad()
            support_ys_pred = NN(support_features, support_ys, support_features)
            loss = -criterion(torch.tensor(support_ys_pred, requires_grad= True, dtype=torch.float64), 
                             torch.tensor(support_ys, requires_grad=True, dtype=torch.float64)
                            )
            loss.backward()
            optimizer.step()

            query_ys_pred = NN(support_features, support_ys, query_features)
        elif classifier == 'Cosine':
            optimizer.zero_grad()
            support_ys_pred = Cosine(support_features, support_ys, support_features)
            loss = -criterion(torch.tensor(support_ys_pred, requires_grad= True, dtype=torch.float64), 
                             torch.tensor(support_ys, requires_grad=True, dtype=torch.float64)
                            )
            loss.backward()
            optimizer.step()
            query_ys_pred = Cosine(support_features, support_ys, query_features)
        else:
            raise NotImplementedError('classifier not supported: {}'.format(classifier))
        
        net.eval()
        loss = criterion(torch.tensor(query_ys_pred, requires_grad= True, dtype=torch.float64), torch.tensor(query_ys, requires_grad=True, dtype=torch.float64))
        losses.append(-loss.item())


        accuracy = metrics.accuracy_score(query_ys, query_ys_pred)
        acc.append(accuracy)
    # print("Loss Mean: %s, Acc Mean: %s" % (np.mean(losses), np.mean(accuracy)))
    torch.cuda.empty_cache()

    return mean_confidence_interval(acc), losses





def meta_test(net, testloader, use_logit=True, is_norm=True, classifier='LR', fewshot_batch_size = 64):
    net = net.eval()
    acc = []

    with torch.no_grad():
        for idx, data in enumerate(tqdm(testloader, ascii = True)):
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            batch_size, _, height, width, channel = support_xs.size()
            support_xs = support_xs.view(-1, height, width, channel)
            query_xs = query_xs.view(-1, height, width, channel)

            if use_logit:
                query_features = net(query_xs).view(query_xs.size(0), -1)
            else:
                feat_query, _ = net(query_xs, is_feat=True)
                query_features = feat_query[-1].view(query_xs.size(0), -1)

            if is_norm:
                query_features = normalize(query_features)


            sup_xs_loader = DataLoader(FewShotBatch(support_xs), batch_size = fewshot_batch_size)

            support_features = []

            for idx_sup, sup in enumerate(sup_xs_loader):
                if use_logit:
                    support_feature = net(sup).view(sup.size(0), -1)
                else:
                    feat_support, _ = net(sup, is_feat=True)
                    support_feature = feat_support[-1].view(sup.size(0), -1)

                if is_norm:
                    support_feature = normalize(support_feature)
                support_features.append(support_feature)

            support_features = torch.cat(support_features, dim = 0)
            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()

            support_ys = support_ys.view(-1).numpy()
            query_ys = query_ys.view(-1).numpy()

            if classifier == 'LR':
                clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000,
                                         multi_class='multinomial')
                clf.fit(support_features, support_ys)
                query_ys_pred = clf.predict(query_features)
            elif classifier == 'NN':
                query_ys_pred = NN(support_features, support_ys, query_features)
            elif classifier == 'Cosine':
                query_ys_pred = Cosine(support_features, support_ys, query_features)
            else:
                raise NotImplementedError('classifier not supported: {}'.format(classifier))

            acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
            torch.cuda.empty_cache()

    return mean_confidence_interval(acc)


def NN(support, support_ys, query):
    """nearest classifier"""
    support = np.expand_dims(support.transpose(), 0)
    query = np.expand_dims(query, 2)

    diff = np.multiply(query - support, query - support)
    distance = diff.sum(1)
    min_idx = np.argmin(distance, axis=1)
    pred = [support_ys[idx] for idx in min_idx]
    return pred


def Cosine(support, support_ys, query):
    """Cosine classifier"""
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    support = support / support_norm
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / query_norm

    cosine_distance = query @ support.transpose()
    max_idx = np.argmax(cosine_distance, axis=1)
    pred = [support_ys[idx] for idx in max_idx]
    return pred
