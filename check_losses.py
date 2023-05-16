import os
import json
import numpy as np
import termplotlib as tpl
import sys

if __name__ == '__main__':
    cutoff = float(sys.argv[1])
    model_dir = sys.argv[2]
    losses = []
    for file_name in os.listdir(f'{model_dir}'):
        if file_name[-4:] != 'json':
            continue
        with open(f'{model_dir}/{file_name}', 'r') as f:
            j = json.load(f)
        loss = float(j['loss'])
        if loss < cutoff:
            losses.append(float(j['loss']))

    counts, bin_edges = np.histogram(losses)

    fig = tpl.figure()
    fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False)
    fig.show()
