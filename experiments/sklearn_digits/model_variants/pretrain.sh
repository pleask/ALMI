#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --time=1-00:00:00
#SBATCH --output=sklearn_digits/outs/slurm-%j.out
#SBATCH --array=0-99
#SBATCH --gres=gpu

source /etc/profile
module load cuda/11.7

WANDB_DISABLED=true stdbuf -oL /home3/wclv88/bounding-mi/bin/python bounding-mi-repo/experiments/sklearn_digits/benchmark.py --seed 0 --device cuda --train_subject_models --subject_model_count 1000 --subject_model_path sklearn_digits/variant_data/subject-models --subject_model_num_classes 10 --subject_model_example_count $EXAMPLE_COUNT --subject_model_variant $((SLURM_ARRAY_TASK_ID))