#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -p shared
#SBATCH --time=2-00:00:00
#SBATCH --output=/nobackup/wclv88/sklearn_digits/outs/slurm-%A_%a.out
#SBATCH --array=0-9999

module load python/3.10.8
module load $PYTHON_BUILD_MODULES

COUNT=10
VARIANT_COUNT=100

WANDB_DISABLED=true \
stdbuf -oL \
python3 \
bounding-mi-repo/experiments/sklearn_digits/benchmark.py \
--seed 0 \
--device cpu \
--train_subject_models \
--subject_model_count 10 \
--subject_model_path /nobackup/wclv88/sklearn_digits/variant_data/subject-models \
--subject_model_num_classes 10 \
--subject_model_variant $((SLURM_ARRAY_TASK_ID % VARIANT_COUNT)) \
--example_start_index $((SLURM_ARRAY_TASK_ID / VARIANT_COUNT * COUNT))