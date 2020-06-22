# ======================
# exampler commands on miniImageNet
# ======================

# supervised pre-training
python train_supervised.py --trial pretrain --model_path /path/to/save --tb_path /path/to/tensorboard --data_root /path/to/data_root
# python train_supervised.py --trial pretrain --model_path path/to/save --tb_path path/to/tensorboard --data_root path/to/data_root


# distillation
# setting '-a 1.0' should give simimlar performance
# python train_distillation.py -r 0.5 -a 0.5 --path_t /path/to/teacher.pth --trial born1 --model_path /path/to/save --tb_path /path/to/tensorboard --data_root /path/to/data_root
python train_distillation.py -r 0.5 -a 0.5 --path_t path/to/save/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_last.pth --trial born1 --model_path path/to/save --tb_path path/to/tensorboard --data_root path/to/data_root
# evaluation
python eval_fewshot.py --model_path /path/to/student.pth --data_root /path/to/data_root
# python eval_fewshot.py --model_path path\to\save\resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain\resnet12_last.pth --data_root path\to\data_root --num_workers 0 --n_shots 1