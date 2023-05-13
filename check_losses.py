import os
import json
import numpy as np
import termplotlib as tpl

losses = []
for file_name in os.listdir('subject_models'):
    if file_name[-4:] != 'json':
        continue
    with open(f'subject_models/{file_name}', 'r') as f:
        j = json.load(f)
    loss = float(j['loss'])
    if loss < 0.00001:
        losses.append(float(j['loss']))

counts, bin_edges = np.histogram(losses)

fig = tpl.figure()
fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False)
fig.show()
