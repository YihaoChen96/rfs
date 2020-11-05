# python eval_fewshot.py --model_path path/to/save/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_last.pth --data_root path/to/data_root --n_shots 10 --n_aug_support_samples 1 --n_ways 10

# python eval_fewshot.py --model_path path/to/save/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_last.pth --data_root path/to/data_root --n_shots 40 --n_aug_support_samples 1 --n_ways 2 --num_workers 0


python eval_fewshot.py --model_path path/to/save/resnet12_tieredImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_last.pth --data_root /srv/datasets/ImageNet --n_shots 2 --n_aug_support_samples 1 --n_ways 160 --dataset tieredImageNet --num_workers 8


