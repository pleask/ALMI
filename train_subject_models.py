import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import argparse
import uuid

from torch.utils.data import DataLoader

from auto_mi.tasks import SimpleFunctionRecoveryTask, VAL
from auto_mi.models import SimpleFunctionRecoveryModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    training_data = [iter(DataLoader(task.get_dataset(i), batch_size=batch_size, num_workers=32)) for i in range(len(nets))]

    parallel_nets = [nn.DataParallel(net) for net in nets]
    optimizers = [optim.Adam(net.parameters(), lr=0.01, weight_decay=weight_decay) for net in nets]

    for epoch in range(epochs):
        if epoch % 1000 == 0:
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
    "--model",
    default="SimpleFunctionRecoveryModel",
    type=str,
    help='Which subject model to use.',
)
parser.add_argument(
    "--epochs", type=int, help="The number of epochs for which to train the models."
)
parser.add_argument(
    "--weight_decay", default=0., type=float, help="Weight decay for the adam optimiser."
)

get_model_path = lambda path, net_idx: f"{path}/{net_idx}.pickle"

if __name__ == "__main__":
    print('Parsing args')
    args = parser.parse_args()
    random.seed(a=args.seed)

    if args.task == "SimpleFunctionRecoveryTask":
        task = SimpleFunctionRecoveryTask(args.count)
    else:
        raise ValueError("Invalid task specified")

    if args.model == "SimpleFunctionRecoveryModel":
        model = SimpleFunctionRecoveryModel
    else:
        raise ValueError("Invalid model specified")

    print("Training...")
    nets = [model(task) for _ in range(args.count)]
    train_subject_nets(nets, task, args.epochs, weight_decay=args.weight_decay)

    print("Evaluating models")
    losses = evaluate_subject_nets(nets, task)

    for i in range(args.count):
        net_id = uuid.uuid4()
        md = vars(args)
        md.update(task.get_dataset(i).get_metadata())
        md["loss"] = losses[i]
        md["id"] = str(net_id)
        with open(f'{args.path}/index.txt', 'a') as md_file:
            md_file.write(json.dumps(md) + '\n')

        model_path = get_model_path(args.path, net_id) 
        torch.save(nets[i].state_dict(), model_path)
