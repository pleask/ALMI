"""
Some of the json metadata files being created aren't valid, this just checks
which ones aren't.
"""
import os
import json

if __name__=='__main__':
    files = [f for f in os.listdir('subject_models') if f[-5:] == '.json']
    for file in files:
        try:
            with open(f'subject_models/{file}', 'r') as j:
                json.load(j)
        except json.JSONDecodeError:
            print(file)
