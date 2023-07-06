declare -a weight_decays=("0" "0.0001" "0.001" "0.01" "0.1" "1")
for weight_decay in "${weight_decays[@]}"
do
    echo $weight_decay
    sbatch train_multifunction_mi_model.sh $weight_decay 0.0
done
