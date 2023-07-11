declare -a weight_decays=("0" "0.0001" "0.001" "0.01" "0.1" "1")
for weight_decay in "${weight_decays[@]}"
do
    echo $function $weight_decay
    sbatch --array=1-10000 train_subject_models.sh $weight_decay
done
