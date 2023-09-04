import pandas as pd
import json
import sys
import matplotlib.pyplot as plt

def flatten_dict(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def read_jsonl_to_dataframe(file_path):
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            flat_record = flatten_dict(record)
            records.append(flat_record)
    return pd.DataFrame(records)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = read_jsonl_to_dataframe(file_path)

    # Create a histogram of the 'loss' column
    plt.hist(df['loss'], bins='auto', alpha=0.7, color='blue', edgecolor='black')
    plt.title('Histogram of Loss Values')
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.show()

    print(type(df['example.operations'][0]))
    df['example.operations'] = df['example.operations'].astype(str)
    print(df[df['example.operations'] == "[[3, '%'], [2, '//'], [0, '+'], [1, '-'], [3, '%']]"])
