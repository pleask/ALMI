import json
import tarfile
import os
import sys

def create_tar_from_index(subject_model_path, tar_file_path):
    index_file_path = f'{subject_model_path}/index.txt'
    # Open the tar archive for writing
    with tarfile.open(tar_file_path, 'w') as tar:
        # Open the index file for reading
        with open(index_file_path, 'r') as f:
            # Iterate over each line in the index file
            for i, line in enumerate(f):
                print(f'Adding file {i}')
                # Parse the line as a JSON object
                data = json.loads(line)

                # Extract the "id" field
                file_id = data['id']

                if len(data['example']['operations']) != 2:
                    continue

                # Construct the filename of the corresponding .pickle file
                pickle_file_path = f"{subject_model_path}/{file_id}.pickle"

                # Check if the file exists
                if os.path.exists(pickle_file_path):
                    # Add the .pickle file to the tar archive
                    tar.add(pickle_file_path)
                else:
                    print(f"Warning: File {pickle_file_path} not found!")

    print(f"Archive {tar_file_path} created successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <index_file_path> <tar_file_path>")
        sys.exit(1)

    index_file_path = sys.argv[1]
    tar_file_path = sys.argv[2]

    create_tar_from_index(index_file_path, tar_file_path)

