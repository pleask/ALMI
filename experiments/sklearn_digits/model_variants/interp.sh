#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --time=2-00:00:00
#SBATCH --output=sklearn_digits/outs/slurm-%j.out
#SBATCH --gres=gpu

source /etc/profile
module load cuda/11.7

NUM_DIGITS=$NUM_DIGITS stdbuf -oL /home3/wclv88/bounding-mi/bin/python /home3/wclv88/bounding-mi-repo/experiments/sklearn_digits/benchmark.py --seed 0 --device cuda --subject_model_path sklearn_digits/variant_data/subject-models --interpretability_model_path=sklearn_digits/interpretability_models --interpretability_model_batch_size 8 --subject_model_num_classes 10 --interpretability_model_embedded --interpretability_model_split_on_variants