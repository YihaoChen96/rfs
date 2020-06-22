#!/bin/sh -l
#SBATCH --gres=gpu:2
#SBATCH -p debug
#SBATCH -c 2
#SBATCH -t 1:00:00
#SBATCH -J train
#SBATCH --signal=USR1@600
#SBATCH --qos=overcap
#SBATCH -o log/train.log
#SBATCH -x asimo,jill,hal,ash

hostname
echo $CUDA_VISIBLE_DEVICES
# srun python eval_fewshot.py --model_path path/to/save/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_last.pth --data_root path/to/data_root --n_shots 40
srun new_run.sh

