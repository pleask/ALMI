#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p cpu
#SBATCH --time=0-06:00:00
#SBATCH -o bounding-mi-data/symbolic/outs/slurm-%A_%a.out
#SBATCH --array 0-9999

source /etc/profile

batch_size=10
start_index=$((SLURM_ARRAY_TASK_ID*batch_size))
end_index=$(((SLURM_ARRAY_TASK_ID+1)*batch_size))

source /etc/profile
stdbuf -oL /home2/wclv88/bounding-mi/bin/python bounding-mi-repo/experiments/symbolic/train_subject.py --path ~/bounding-mi-data/symbolic/subject-models --seed 0 --start_idx $start_index --end_idx $end_index
