#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --time=2-00:00:00
epochs=1000

source /etc/profile
module load cuda/11.7

source /home2/wclv88/bounding-mi/bounding-mi/bin/activate

# addition 		b03c95f7-2f95-4e68-8726-11061097052d
# multiplication 	8d7bb88a-f542-4d46-9134-1c9b7541e7f3
# sigmoid        	256f187a-cf1d-4ad2-acaa-152c4018e0b6
# exponent 		9c619c5a-90a7-4931-868f-ef1e97a67030
# min 			ba2f9150-1bc8-4b19-9186-306dbc98ee86
stdbuf -oL /home2/wclv88/bounding-mi/bounding-mi/bin/python train_multifunction_mi_model.py --model_folders experiments/b03c95f7-2f95-4e68-8726-11061097052d/subject_models experiments/8d7bb88a-f542-4d46-9134-1c9b7541e7f3/subject_models experiments/256f187a-cf1d-4ad2-acaa-152c4018e0b6/subject_models experiments/9c619c5a-90a7-4931-868f-ef1e97a67030/subject_models experiments/ba2f9150-1bc8-4b19-9186-306dbc98ee86/subject_models --epochs 1000 --model_path multi.pickle
