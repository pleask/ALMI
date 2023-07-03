import pandas as pd
import json 

with open('index.txt', 'r') as index_file:
    rows = []
    for line in index_file:
        line = line.strip()
        model_name, metadata_string = line.split(' ', maxsplit=1)
        metadata = json.loads(metadata_string)
        rows.append(metadata)

df = pd.DataFrame(rows)
