declare -a functions=("addition" "multiplication" "sigmoid" "exponent" "min")
declare -a weight_decays=("0" "0.0001" "0.001" "0.01" "0.1" "1")
for function in "${functions[@]}"
do
    for weight_decay in "${weight_decays[@]}"
    do
        echo $function $weight_decay
        sbatch --array=0-2000 train_subject_models.sh $function $weight_decay
    done
done

