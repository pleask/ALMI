#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p shared
#SBATCH --time=0-06:00:00
#SBATCH -o /nobackup/wclv88/bounding-mi-data/integer/outs/slurm-%A_%a.out
#SBATCH --array 0-9999

source /etc/profile

module load python/3.10.8
module load $PYTHON_BUILD_MODULES
# pip install /home/wclv88/bounding-mi-repo/auto_mi

batch_size=10
start_index=$((SLURM_ARRAY_TASK_ID*batch_size))
end_index=$(((SLURM_ARRAY_TASK_ID+1)*batch_size))

source /etc/profile
stdbuf -oL python3 bounding-mi-repo/experiments/integer/train_subject.py --path /nobackup/wclv88/bounding-mi-data/integer/subject-models --seed 0 --start_idx $start_index --end_idx $end_index
