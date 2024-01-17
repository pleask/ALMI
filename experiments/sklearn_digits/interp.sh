#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --time=2-00:00:00
#SBATCH --output=sklearn_digits/outs/slurm-%j.out
#SBATCH --array=0-5
#SBATCH --gres=gpu:ampere:1

source /etc/profile
module load cuda/11.7

NUM_DIGITS=$((5 + SLURM_ARRAY_TASK_ID))

NUM_DIGITS=$NUM_DIGITS stdbuf -oL /home3/wclv88/bounding-mi/bin/python /home3/wclv88/bounding-mi-repo/experiments/sklearn_digits/benchmark.py --seed 0 --device cuda --subject_model_path sklearn_digits/${NUM_DIGITS}/subject-models --interpretability_model_path=sklearn_digits/interpretability_models --interpretability_model_batch_size 8 --interpretability_model_embedded