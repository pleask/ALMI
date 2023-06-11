#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --time=2-00:00:00
#SBATCH --output=mi_models_out/%j.txt
epochs=10
weight_decay=$1

source /etc/profile
module load cuda/11.7

source /home2/wclv88/bounding-mi/bounding-mi/bin/activate

echo "${weight_decay}"

stdbuf -oL /home2/wclv88/bounding-mi/bounding-mi/bin/python train_multifunction_mi_model.py --epochs $epochs --subject_model_dir subject_models/ --weight_decay $weight_decay --max_loss 0.001 --model_path "mi_models/${weight_decay}.pickle"
