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

source /etc/profile

for index in $(seq 0 19); do
	stdbuf -oL python3 bounding-mi-repo/experiments/integer/pipeline.py --pretrain --subject_model_path /nobackup/wclv88/bounding-mi-data/integer/subject-models
done
