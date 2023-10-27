#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p shared
#SBATCH --time=2-00:00:00
#SBATCH -o /nobackup/wclv88/bounding-mi-data/mnist/outs/slurm-%A_%a.out
#SBATCH --array 0-4999

module load python/3.10.8
module load $PYTHON_BUILD_MODULES

SEED=$SLURM_ARRAY_TASK_ID
stdbuf -oL bounding-mi/bin/python bounding-mi-repo/experiments/mnist_permuted/benchmark.py --seed $SEED