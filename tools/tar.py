import json
import tarfile
import os

# Define the path to the index file and the output tar archive
index_file_path = 'index.txt'
tar_file_path = 'archive.tar'

# Open the tar archive for writing
with tarfile.open(tar_file_path, 'w') as tar:
    # Open the index file for reading
    with open(index_file_path, 'r') as f:
        # Iterate over each line in the index file
        for line in f:
            # Parse the line as a JSON object
            data = json.loads(line)
            
            # Extract the "id" field
            file_id = data['id']
            
            # Construct the filename of the corresponding .pickle file
            pickle_file_path = f"{file_id}.pickle"
            
            # Check if the file exists
            if os.path.exists(pickle_file_path):
                # Add the .pickle file to the tar archive
                tar.add(pickle_file_path)
            else:
                print(f"Warning: File {pickle_file_path} not found!")

print(f"Archive {tar_file_path} created successfully!")
