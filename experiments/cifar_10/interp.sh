#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name interp
#SBATCH --ntasks 20
#SBATCH --output=experiments/cifar_10/slurm_out/%x.txt
#SBATCH --gres gpu

source almi_env/bin/activate

NUM_DIGITS=$((5 + SLURM_ARRAY_TASK_ID))

NUM_DIGITS=$NUM_DIGITS WANDB_DISABLED=False python3 experiments/cifar_10/benchmark.py \
--seed 0 \
--device cuda \
--subject_model_path experiments/cifar_10/${NUM_DIGITS}/subject-models \
--interpretability_model_path experiments/cifar_10/interpretability_models \
--interpretability_model_batch_size 8 \
--subject_model_num_classes $NUM_DIGITS \
