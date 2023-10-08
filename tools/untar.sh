#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --qos=short
#SBATCH -p cpu
#SBATCH --array=0-255
#SBATCH --time=01:00:00

# Convert the array index into a two digit hexadecimal number
FILE_NUM=$(printf "%02x" $SLURM_ARRAY_TASK_ID)

# Extract the corresponding tar archive
tar -xf ${FILE_NUM}.tar

echo "Extraction of archive ${FILE_NUM}.tar is complete."
