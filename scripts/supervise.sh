# ======================
# exampler commands on miniImageNet
# ======================

# supervised eval
python eval_supervised.py --trial supervised --model_path /path/to/save/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_last.pth --tb_path /path/to/tensorboard --data_root /path/to/data_root