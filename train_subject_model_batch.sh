#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --time=0-00:10:00

source /etc/profile
module load cuda/11.7

source /home2/wclv88/bounding-mi/bin/activate
stdbuf -oL /home2/wclv88/bounding-mi/bounding-mi/bin/python train_subject_models.py subject_models $1 5 $1 addition 10000
