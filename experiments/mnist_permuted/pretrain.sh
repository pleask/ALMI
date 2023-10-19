#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p shared
#SBATCH --time=0-06:00:00
#SBATCH --output=mnist/outs/%A_%a.out
#SBATCH --array 0-9999

source /etc/profile
module load cuda/11.7

source /etc/profile

for index in $(seq 0 4); do
	stdbuf -oL python3 bounding-mi-repo/experiments/mnist_perturbed/benchmark.py
done
