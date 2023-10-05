#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p shared
#SBATCH --time=0-06:00:00
#SBATCH -o /nobackup/wclv88/bounding-mi-data/integer/outs/slurm-%A_%a.out
#SBATCH --array 0-999

source /etc/profile

module load python/3.10.8
module load $PYTHON_BUILD_MODULES
pip install /home/wclv88/bounding-mi-repo/auto_mi

source /etc/profile

# Check if SLURM_ARRAY_TASK_ID is 0 to execute a different script
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    stdbuf -oL python3 bounding-mi-repo/tools/tar.py /nobackup/wclv88/bounding-mi-data/integer/subject-models /nobackup/wclv88/bounding-mi-data/integer/subject-models.tar
else
    stdbuf -oL python3 bounding-mi-repo/experiments/integer/pipeline.py --pretrain
fi