# TODO: unify with tar ? parallelise? support filtering?
import sys
import json
import os

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <index_file_path>")
        sys.exit(1)

    subject_model_path = sys.argv[1]
    index_file_path = f'{subject_model_path}/index.txt'
    # Open the index file for reading
    with open(index_file_path, 'r') as f:
        # Iterate over each line in the index file
        for i, line in enumerate(f):
            print(f'Deleting file {i}')
            # Parse the line as a JSON object
            data = json.loads(line)

            # Extract the "id" field
            file_id = data['id']

            # Construct the filename of the corresponding .pickle file
            pickle_file_path = f"{subject_model_path}/{file_id}.pickle"

            # Check if the file exists
            if os.path.exists(pickle_file_path):
                os.remove(pickle_file_path)
            else:
                print(f"Warning: File {pickle_file_path} not found!")
