"""
Trains subject models, ie. the models that implement the labeling function.

Takes exactly three arguments:
- path: directory to which to save the models.
- start: from which model index to start training.
- count: the number of models to train in parallel on the single GPU. Currently
  think this should be around 5, but this might depend on the GPU.
- seed: the random seed to use. Should be different for all runs of this script
  within an experiment, and the same for each run across experiments.
- fn: the function to train the subject nets on. Eg. 'addition' for addition.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import random
import json
from functools import partial

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
The layer size of the subject networks.

Ran a grid search for layer size with 5000 epochs, with the following results
Layer size 20 with 481 total parameters achieved min loss of 0.37094342708587646
Layer size 25 with 726 total parameters achieved min loss of 0.11320501565933228
Layer size 30 with 1021 total parameters achieved min loss of 0.18821430206298828
Layer size 35 with 1366 total parameters achieved min loss of 0.12562520802021027
Layer size 40 with 1761 total parameters achieved min loss of 0.13527408242225647
Layer size 45 with 2206 total parameters achieved min loss of 0.07131229341030121
Layer size 50 with 2701 total parameters achieved min loss of 0.04721994698047638
Layer size 100 with 10401 total parameters achieved min loss of 0.05302942544221878

25 offers decent performance whilst being a third the size of the first larger
network that performs better
"""
SUBJECT_LAYER_SIZE = 25
"""
The number of epochs for training the subject networks.

Ran a grid search for epochs on LAYER_SIZE = 25 with the following results
200 epochs achieved avg loss of 15.250823020935059
500 epochs achieved avg loss of 6.639222621917725
1000 epochs achieved avg loss of 1.6066440343856812
2000 epochs achieved avg loss of 2.3655612468719482
5000 epochs achieved avg loss of 0.7449145317077637
10000 epochs achieved avg loss of 0.28002771735191345
Also ran this for 20k epochs and the performance was not better than 10k
"""
SUBJECT_EPOCHS = 10000
"""
The batch size when training the subject neworks.

This is nowhere the limit for the batch size in terms of GPU memory, but the
difference in training times between this batch size and the maximum possible
batch size was small, and this smaller batch size means we can train a number
of networks in parallel on the same GPU.
"""
SUBJECT_BATCH_SIZE = 2**15
SUBJECT_CRITERION = nn.MSELoss()
"""
Absolutely no idea how many we will need here, suggest starting with 10k and
going from there. Using 100 seemed to result in the mi model not learning, but
on that run the subject models had only been trained for 100 epochs rather than
10k so it's possible there just wasn't any signal to learn.
"""
TOTAL_SUBJECT_MODELS = 100
# Path at which to store the subject models
SUBJECT_MODEL_PATH = "./subject_models"


get_exponent = lambda: random.random() * 10.0


def get_subject_data(fn, batch_count=2**10):
    """
    Generate random data on the GPU for training the subject networks.
    """
    x = torch.rand((batch_count, SUBJECT_BATCH_SIZE), device=DEVICE) * 10.0
    x = torch.unsqueeze(x, dim=1)
    y = fn(x)
    data = torch.empty((x.shape[0], 2, x.shape[2]), device=DEVICE, dtype=torch.float32)
    torch.cat((x, y), dim=1, out=data)
    data = torch.unsqueeze(data, dim=3)
    return data


def get_subject_net():
    """
    Returns an instance of a subject network. The layer sizes have been tuned
    by grid search, but the general architecture of the network has not been
    experimented with.
    """
    return nn.Sequential(
        nn.Linear(1, SUBJECT_LAYER_SIZE),
        nn.ReLU(),
        nn.Linear(SUBJECT_LAYER_SIZE, SUBJECT_LAYER_SIZE),
        nn.ReLU(),
        nn.Linear(SUBJECT_LAYER_SIZE, 1),
    ).to(DEVICE)


def train_subject_nets(nets, fns):
    """
    Trains subject networks in parallel. From brief testing it seems 5 subject nets
    can be trained in parallel, but it's up to the client to check this.
    """
    optimizers = [optim.Adam(net.parameters(), lr=0.01) for net in nets]
    training_data = [get_subject_data(fn) for fn in fns]
    parallel_nets = [nn.DataParallel(net) for net in nets]
    for epoch in range(SUBJECT_EPOCHS):
        if epoch % 1000 == 0:
            print(f'Epoch {epoch} of {SUBJECT_EPOCHS}', flush=True)
        for batch_idx in range(len(training_data)):
            for net, data, optimizer in zip(parallel_nets, training_data, optimizers):
                optimizer.zero_grad()
                inputs, labels = data[batch_idx]
                output = net(inputs)
                loss = SUBJECT_CRITERION(output, labels)
                loss.backward()
                optimizer.step()


def evaluate_subject_nets(nets, fns):
    """
    Evaluates the subject net using a new random dataset.
    """
    losses = []
    for net, fn in zip(nets, fns):
        eval_data = get_subject_data(fn, batch_count=1)[0]
        inputs, labels = eval_data
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs)
        loss = SUBJECT_CRITERION(outputs, labels).detach().cpu().item()
        losses.append(loss)
    return losses


get_subject_model_path = lambda i: f"{SUBJECT_MODEL_PATH}/{i}.pickle"
get_subject_model_metadata_path = lambda i: f"{SUBJECT_MODEL_PATH}/{i}_metadata.json"


# TODO: Introduce more functions as per notes
# TODO: This would probably be nicer if it returned classes that are instantiated with the parameters
def get_subject_fn(fn_name, *params):
    if fn_name == 'addition':
        return partial(lambda c, x: x + c, params[0])
    elif fn_name == 'multiplication':
        return partial(lambda c, x: x * c, params[0])
    raise ValueError('Invalid function name')


if __name__ == '__main__':
    path = sys.argv[1]
    start = int(sys.argv[2])
    count = int(sys.argv[3])
    seed = int(sys.argv[4])
    fn_name = sys.argv[5]
    random.seed(a=seed)

    print(f'Training {count} models from index {start}')
    nets = [get_subject_net() for _ in range(count)]
    fns = [get_subject_fn(fn_name, random.random()) for _ in range(count)]
    train_subject_nets(nets, fns)

    print('Evaluating models')
    losses = evaluate_subject_nets(nets, fns)
    metadata = [{
        "exponent": exponent,
        "loss": loss
    } for exponent, loss in zip(fns, losses)]

    print('Saving models')
    for i, (net, md) in enumerate(zip(nets, metadata)):
        net_idx = start + i
        model_path = f'{path}/{net_idx}.pickle'
        md_path = f'{path}/{net_idx}_metadata.json'
        torch.save(net.state_dict(), model_path)
        with open(md_path, "w") as json_file:
            json.dump(md, json_file)
