#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --time=2-00:00:00
#SBATCH --output=integer/outs/%j.out

source /etc/profile
module load cuda/11.7

stdbuf -oL /home2/wclv88/bounding-mi/bin/python /home2/wclv88/bounding-mi-repo/experiments/integer/pipeline.py