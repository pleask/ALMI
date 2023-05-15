#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --time=0-00:10:00
#SBATCH --array=1-2000
#SBATCH -o experiments/%A/outs/%a.out

f=addition
epochs=10000

dir=experiments/$SLURM_JOB_ID
mkdir $dir
mkdir $dir/outs
mkdir $dir/subject_models
echo "function: $f epochs: $epochs" >> $SLURM_JOB_ID/metadata.txt

source /etc/profile
module load cuda/11.7
i=$((5*$SLURM_ARRAY_TASK_ID+1))
source /home2/wclv88/bounding-mi/bounding-mi/bin/activate
stdbuf -oL /home2/wclv88/bounding-mi/bounding-mi/bin/python train_subject_models.py --path=$dir/subject_models --start=$i --count=5 --seed=$i --fn_name=$f --epochs=$epochs
