#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --time=0-00:10:00
#SBATCH --array=400-2000
#SBATCH -o outs/slurm-%A_%a.out

source /etc/profile
module load cuda/11.7
i=$((5*$SLURM_ARRAY_TASK_ID+1))
source /home2/wclv88/bounding-mi/bounding-mi/bin/activate
stdbuf -oL /home2/wclv88/bounding-mi/bounding-mi/bin/python train_subject_models.py subject_models $i 5 $i addition 100
