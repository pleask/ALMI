#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --time=2-00:00:00
#SBATCH --output=sklearn_digits/outs/slurm-%j.out
#SBATCH --array 0-65

source /etc/profile
module load cuda/11.7

SEED=$SLURM_ARRAY_TASK_ID
stdbuf -oL python3 bounding-mi-repo/experiments/sklearn_digits/benchmark.py --seed $SEED --device cpu --train_subject_models --subject_model_path  --batch_size 100

#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --time=2-00:00:00
#SBATCH --output=sklearn_digits/outs/slurm-%j.out
#SBATCH --array=0-65

source /etc/profile
module load cuda/11.7

# Calculate FROZEN and NUM_DIGITS from SLURM_ARRAY_TASK_ID
FROZEN=$((SLURM_ARRAY_TASK_ID % 11))
NUM_DIGITS=$((5 + SLURM_ARRAY_TASK_ID / 11))

stdbuf -oL FROZEN=$FROZEN NUM_DIGITS=$NUM_DIGITS WANDB_DISABLED=true python3 bounding-mi-repo/experiments/sklearn_digits/benchmark.py --seed 0 --device cuda --train_subject_models --subject_model_count 1000 --subject_model_path /nobackup/wclv88/sklearn_digits/${NUM_DIGITS}/subject-models 