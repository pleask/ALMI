#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --time=2-00:00:00
#SBATCH --output=sklearn_digits/outs/slurm-%j.out
#SBATCH --gres=gpu:ampere:1

source /etc/profile
module load cuda/11.7

WAND_API_KEY=dd685d7aa9b38a2289e5784a961b81e22fc4c735  stdbuf -oL /home2/wclv88/bounding-mi/bin/python /home2/wclv88/bounding-mi-repo/experiments/sklearn_digits/benchmark.py --seed 0 --device cuda --subject_model_path sklearn_digits/subject_models --interpretability_model_path=sklearn_digits/interpretability_models