job_name=$(uuidgen)
dir=experiments/$job_name
mkdir $dir
mkdir $dir/outs
mkdir $dir/subject_models

f='addition'
epochs='epochs'
echo "date: $(date) function: $f epochs: $epochs" >> $dir/metadata.txt

sbatch --job-name=$job_name train_subject_models.sh $f $epochs