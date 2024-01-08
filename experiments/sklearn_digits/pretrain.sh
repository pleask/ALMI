#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --time=1-00:00:00
#SBATCH --output=sklearn_digits/outs/slurm-%j.out
#SBATCH --array=0-65
#SBATCH --gres=gpu

source /etc/profile
module load cuda/11.7

# Calculate FROZEN and NUM_DIGITS from SLURM_ARRAY_TASK_ID
FROZEN=$((SLURM_ARRAY_TASK_ID % 11))
NUM_DIGITS=$((5 + SLURM_ARRAY_TASK_ID / 11))

if [ $FROZEN -eq 10 ]; then
    SUBJECT_MODEL_COUNT=10000
else
    SUBJECT_MODEL_COUNT=1000
fi

FROZEN=$FROZEN NUM_DIGITS=$NUM_DIGITS WANDB_DISABLED=true stdbuf -oL /home3/wclv88/bounding-mi/bin/python bounding-mi-repo/experiments/sklearn_digits/benchmark.py --seed 0 --device cuda --train_subject_models --subject_model_count $SUBJECT_MODEL_COUNT --subject_model_path sklearn_digits/${NUM_DIGITS}/subject-models 