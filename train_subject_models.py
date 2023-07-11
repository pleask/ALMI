import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import argparse
import uuid

from torch.utils.data import DataLoader

from tasks import SimpleFunctionRecoveryTask, VAL

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


def train_subject_nets(nets, task, epochs, batch_size=1000, weight_decay=0):
    """
    Trains subject networks in parallel. From brief testing it seems 5 subject nets
    can be trained in parallel, but it's up to the client to check this.

    epochs: The number of epochs for training the subject networks.
    Ran a grid search for epochs on LAYER_SIZE = 25 with the following results
    200 epochs achieved avg loss of 15.250823020935059
    500 epochs achieved avg loss of 6.639222621917725
    1000 epochs achieved avg loss of 1.6066440343856812
    2000 epochs achieved avg loss of 2.3655612468719482
    5000 epochs achieved avg loss of 0.7449145317077637
    10000 epochs achieved avg loss of 0.28002771735191345
    Also ran this for 20k epochs and the performance was not better than 10k
    """
    training_data = [iter(DataLoader(task.get_dataset(i), batch_size=batch_size)) for i in range(len(nets))]

    parallel_nets = [nn.DataParallel(net) for net in nets]
    optimizers = [optim.Adam(net.parameters(), lr=0.01, weight_decay=weight_decay) for net in nets]

    for epoch in range(epochs):
        # if epoch % 1000 == 0:
        print(f"Epoch {epoch} of {epochs}", flush=True)
        while True:
            try:
                for net, data, optimizer in zip(parallel_nets, training_data, optimizers):
                    optimizer.zero_grad()
                    inputs, labels = next(data)
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    output = net(inputs)
                    loss = task.criterion(output, labels)
                    loss.backward()
                    optimizer.step()
            except StopIteration:
                break


def evaluate_subject_nets(nets, task, batch_size=1000):
    """
    Evaluates the subject net using a new random dataset.
    """
    losses = []
    examples = [task.get_dataset(i, type=VAL) for i in range(len(nets))]
    for net, example in zip(nets, examples):
        data = DataLoader(example, batch_size=batch_size)
        inputs, labels = next(iter(data))
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs)
        loss = task.criterion(outputs, labels).detach().cpu().item()
        losses.append(loss)

        print(f"SAMPLE PREDICTIONS (label, prediction) for {example.get_metadata()}", flush=True)
        [
            print(l.detach().cpu().item(), o.detach().cpu().item(), flush=True)
            for l, o in zip(labels[:10], outputs[:10])
        ]
        print()

    return losses


parser = argparse.ArgumentParser(
    description="Trains subject models, ie. the models that implement the labeling function."
)
parser.add_argument("--path", type=str, help="Directory to which to save the models")
parser.add_argument(
    "--count",
    type=int,
    help="The number of models to train in parallel on the single GPU. Currently think this should be around 5, but this might depend on the GPU.",
    required=False,
)
parser.add_argument(
    "--seed",
    type=int,
    help="The random seed to use. Should be different for all runs of this script within an experiment, and the same for each run across experiments.",
)
parser.add_argument(
    "--task",
    default="SimpleFunctionRecoveryTask",
    type=str,
    help='The task on which to train the subject models.',
)
parser.add_argument(
    "--epochs", type=int, help="The number of epochs for which to train the models."
)
parser.add_argument(
    "--weight_decay", default=0., type=float, help="Weight decay for the adam optimiser."
)

get_model_path = lambda path, net_idx: f"{path}/{net_idx}.pickle"
get_metadata_path = lambda path, net_idx: f"{path}/{net_idx}_metadata.json"

if __name__ == "__main__":
    print('Parsing args')
    args = parser.parse_args()
    random.seed(a=args.seed)

    task = SimpleFunctionRecoveryTask(args.count)
    if args.task != "SimpleFunctionRecoveryTask":
        raise ValueError("Invalid task specified")

    print("Training...")
    nets = [get_subject_net() for _ in range(args.count)]
    train_subject_nets(nets, task, args.epochs, weight_decay=args.weight_decay)

    print("Evaluating models")
    losses = evaluate_subject_nets(nets, task)

    metadata = []
    for i in range(args.count):
        md = vars(args)
        md.update(task.get_dataset(i).get_metadata())
        md["loss"] = losses[i]
        metadata.append(md)

    print("Saving models")
    for i, (net, md) in enumerate(zip(nets, metadata)):
        net_id = uuid.uuid4()
        model_path = get_model_path(args.path, net_id) 
        md_path = get_metadata_path(args.path, net_id)
        torch.save(net.state_dict(), model_path)
        with open(md_path, "w") as json_file:
            json.dump(md, json_file)
