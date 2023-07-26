#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --time=2-00:00:00
#SBATCH --output=mi_models_out/%j.out
#SBATCH --array=0-10
epochs=1000

weight_decays=("0." "0.0001" "0.001" "0.01" "0.1" "1")
prune_amounts=("0." "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
repeats=3

weight_decay_queue=()
prune_amount_queue=()
for ((i = 0; i < ${#weight_decays[@]}; i++)); do
    for ((j = 0; j < ${#prune_amounts[@]}; j++)); do
        for ((k = 0; k < N; k++)); do
            weight_decay_queue+=${weight_decays[i]}
            prune_amount_queue+=${prune_amount[j]}
        done
    done
done

weight_decay=${weight_decay_queue[$SLURM_ARRAY_TASK_ID]}
prune_amount=${prune_amount_queue[$SLURM_ARRAY_TASK_ID]}

source /etc/profile
module load cuda/11.7

stdbuf -oL /home2/wclv88/bounding-mi/bin/python train_multifunction_mi_model.py --subject_model_dir bounding-mi-data/symbolic/subject-models/ --weight_decay $weight_decay --prune_amount $prune_amount --model_path "bounding-mi-data/symbolic/mi_models/${weight_decay}.pickle" --seed $SLURM_ARRAY_TASK_ID
