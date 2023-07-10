#!/bin/bash
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -p cpu
#SBATCH --output=subject_models/index.txt

# Create an index file from the metadata files
# Function to process a single file
process_file() {
    file="$1"
    filename=$(basename "$file" _metadata.json)
    contents=$(cat "$file")
    echo "$filename $contents"
}

export -f process_file

# Find all JSON files and pass them to xargs for parallel processing
find . -name '*_metadata.json' -print0 | xargs -0 -I{} -P $(nproc) bash -c 'process_file "{}" &'

# Wait for all background processes to finish
wait
