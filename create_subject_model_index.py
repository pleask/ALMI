"""
Opening the metadata files individually takes absolutely ages on the NCC as the IO is so slow. This script creates an index file for the metadata to make dataset loading faster.
"""
import asyncio
import json
import os

from train_multifunction_mi_model import get_subject_model_metadata

SUBJECT_MODEL_DIR = 'subject_models'
INDEX_FILE = 'subject_models_index.txt'

async def get_metadata():
    all_subject_model_filenames = os.listdir(SUBJECT_MODEL_DIR)
    
    async def get_metadata(filename):
        if not filename.endswith('.pickle'):
            return None
        subject_model_name = filename.removesuffix('.pickle')
        try:
            metadata = get_subject_model_metadata(SUBJECT_MODEL_DIR, subject_model_name)
        except FileNotFoundError:
            print(f'did not find metadata file for model {subject_model_name}')
            return None
        return (subject_model_name, json.dumps(metadata))
        
    tasks = [get_metadata(f) for f in all_subject_model_filenames]
    model_metadata = await asyncio.gather(*tasks)
    return [m for m in model_metadata if m is not None]

if __name__ == '__main__':
    model_metadata = asyncio.run(get_metadata())

    with open(INDEX_FILE, 'w') as file:
        for l in model_metadata:
            file.write(f'{l[0]} {l[1]}\n')
