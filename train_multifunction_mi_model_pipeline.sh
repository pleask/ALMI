declare -a weight_decays=("0" "0.0001" "0.001" "0.01" "0.1" "1")
for weight_decay in "${weight_decays[@]}"
do
    sbatch train_subject_models.sh $weight_decay
done