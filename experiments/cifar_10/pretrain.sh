#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --time=1-00:00:00
#SBATCH --output=cifar_10/outs/slurm-%j.out
#SBATCH --array=0-999
#SBATCH --gres=gpu

source /etc/profile
module load cuda/11.7

WANDB_DISABLED=true stdbuf -oL /home2/zwjh86/zwjh86tune/bin/python almi/experiments/cifar_10/benchmark.py --seed 0 --device cuda --train_subject_models --subject_model_count 50 --subject_model_path cifar_10/subject-models 