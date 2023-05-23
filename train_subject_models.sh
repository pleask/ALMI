#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --time=0-00:10:00
#SBATCH --array=0-0
#SBATCH -o subject_model_outs/%A_%a.out

f=$1
epochs=10000
weight_decay=$2
dir=subject_models

source /etc/profile
module load cuda/11.7
i=$((5*$SLURM_ARRAY_TASK_ID+1))
source /home2/wclv88/bounding-mi/bounding-mi/bin/activate
echo $f $epochs $weight_decay
stdbuf -oL /home2/wclv88/bounding-mi/bounding-mi/bin/python train_subject_models.py --path=$dir --count=5 --seed=$i --fn_name=$f --epochs=$epochs --weight_decay=$weight_decay
