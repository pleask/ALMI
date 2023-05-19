import os
import sys

if __name__ == '__main__':
    subject_model_dir = sys.argv[1]
    files = os.listdir(subject_model_dir)
    missing_files = []
    for i in range(1, len(files) // 2 + 1):
        file_string = f'{i}.pickle'
        if file_string not in files and (i - 1) % 5 == 0:
            missing_files.append((i-1) // 5)
    if len(missing_files) > 0:
        print('Re-run the following batches')
        [print(f) for f in missing_files]
