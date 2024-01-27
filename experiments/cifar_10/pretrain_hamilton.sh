#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -p shared
#SBATCH --time=2-00:00:00
#SBATCH -o /nobackup/wclv88/cifar_10/outs/slurm-%A_%a.out
#SBATCH --array 0-1999

module load python/3.10.8
module load $PYTHON_BUILD_MODULES

SEED=$SLURM_ARRAY_TASK_ID
stdbuf -oL python3 \
    bounding-mi-repo/experiments/cifar_10/benchmark.py \
    --seed $SEED \
    --device cpu \
    --subject_model_path /nobackup/wclv88/cifar_10/subject-models \
 .  --train_subject_models \
    --subject_model_count 5 \
    --subject_model_num_classes 10
