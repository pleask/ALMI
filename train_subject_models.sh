#!/bin/bash
for i in {0..0}
do
  argument=$((5*$i+1))
  sbatch --job-name=$i --output=outs/$i.out train_subject_model_batch.sh $argument
done
