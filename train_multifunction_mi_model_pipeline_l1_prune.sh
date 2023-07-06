declare -a prune_amounts=("0.1" "0.3" "0.4" "0.5" "0.6" )
for prune_amount in "${prune_amounts[@]}"
do
    echo $prune_amount
    sbatch train_multifunction_mi_model.sh 0.0 $prune_amount
done
