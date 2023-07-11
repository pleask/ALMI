#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p cpu
#SBATCH --time=0-00:10:00
#SBATCH -o subject_models_2_outs/%A_%a.out

epochs=5
weight_decay=$1
dir=subject_models_2

source /etc/profile
echo $epochs $weight_decay
stdbuf -oL /home2/wclv88/bounding-mi/bin/python train_subject_models.py --path=$dir --index=$SLURM_ARRAY_TASK_ID --seed=0 --epochs=$epochs --weight_decay=$weight_decay
