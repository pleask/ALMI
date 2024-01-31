#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --time=1-00:00:00
#SBATCH --output=sklearn_digits/constrained_data/outs/slurm-%j.out
#SBATCH --array=0-65
#SBATCH --gres=gpu

source /etc/profile
module load cuda/11.7

EXAMPLE_COUNT_GROUP=$((SLURM_ARRAY_TASK_ID / 11))

case $EXAMPLE_COUNT_GROUP in
    0) EXAMPLE_COUNT=10;;
    1) EXAMPLE_COUNT=50;;
    2) EXAMPLE_COUNT=100;;
    3) EXAMPLE_COUNT=500;;
    4) EXAMPLE_COUNT=1000;;
    5) EXAMPLE_COUNT=-1;;
    *) echo "Invalid group" $EXAMPLE_COUNT; exit 1;;
esac

WANDB_DISABLED=true stdbuf -oL /home3/wclv88/bounding-mi/bin/python \
bounding-mi-repo/experiments/sklearn_digits/benchmark.py \
--seed 0 --seed $SLURM_ARRAY_TASK_ID \
--device cuda \
--train_subject_models \
--subject_model_count 10000 \
--subject_model_path sklearn_digits/constrained_data/subject-models \
--subject_model_num_classes 10 \
--subject_model_example_count $EXAMPLE_COUNT