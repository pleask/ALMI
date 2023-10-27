#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --time=2-00:00:00
#SBATCH --output=mnist/outs/%A_%a.out
#SBATCH --array 0-999
#SBATCH --exclude gpu4,gpu5,gpu6

source /etc/profile
module load cuda/11.7

source /etc/profile

for index in $(seq 0 4); do
    SEED=$(( $SLURM_ARRAY_TASK_ID * 5 + $index ))
	stdbuf -oL bounding-mi/bin/python bounding-mi-repo/experiments/mnist_permuted/benchmark.py --seed $SEED
done