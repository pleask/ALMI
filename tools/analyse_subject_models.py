import pandas as pd
import numpy as np
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


def render_interaction_heatmap(df):
    def extract_second_element(list_of_lists):
        return [inner_list[1] for inner_list in list_of_lists]

    # Use apply to create new columns
    df['op1'], df['op2'] = zip(*df['example.operations'].apply(lambda x: extract_second_element(x)))
    grouped = df.groupby(['op1', 'op2'])['loss'].mean().reset_index()
    pivot_df = grouped.pivot(index='op1', columns='op2', values='loss')
    # Create the heatmap
    fig, ax = plt.subplots()

    # Plot the values using matshow
    cax = ax.matshow(pivot_df, cmap='coolwarm')

    # Add colorbar for reference
    fig.colorbar(cax)

    # Set axis labels
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))

    # Label the axis ticks
    ax.set_xticklabels(pivot_df.columns)
    ax.set_yticklabels(pivot_df.index)

    # Loop over data dimensions and create text annotations.
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            value = pivot_df.iloc[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="w")

    plt.show()


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

    render_interaction_heatmap(df)

