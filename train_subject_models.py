import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import argparse
import uuid
from time import gmtime, strftime

from torch.utils.data import DataLoader

from auto_mi.tasks import VAL, get_task_class
from auto_mi.models import get_subject_model_class


# cpu actually seems to be faster for these small models (perhaps as less frequent transfers to gpu)
DEVICE = torch.device("cpu")


def train_subject_net(net, example, epochs=50, batch_size=1000, weight_decay=0):
    """
    Trains subject networks in parallel. From brief testing it seems 5 subject nets
    can be trained in parallel, but it's up to the client to check this.
    """

    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=weight_decay)
    training_data = DataLoader(example, batch_size=batch_size)

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        for inputs, labels in training_data:
                optimizer.zero_grad()
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                output = net(inputs)
                loss = task.criterion(output, labels)
                loss.backward()
                optimizer.step()


def evaluate_subject_net(net, example, batch_size=1000):
    """
    Evaluates the subject net using a new random dataset.
    """
    data = DataLoader(example, batch_size=batch_size)
    inputs, labels = next(iter(data))
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)
    net.eval()
    with torch.no_grad():
        outputs = net(inputs)
    return task.criterion(outputs, labels).detach().cpu().item()


parser = argparse.ArgumentParser(
    description="Trains subject models, ie. the models that implement the labeling function."
)
parser.add_argument("--path", type=str, help="Directory to which to save the models")
parser.add_argument(
    "--seed",
    type=int,
    help="The random seed to use. Should be different for all runs of this script within an experiment, and the same for each run across experiments.",
)
parser.add_argument(
    "--index",
    type=int,
    help="The index of the subject model.",
    required=True
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


def assert_is_unique_model(path, seed, index, task, model, epochs, weight_decay):
    try:
        with open(f'{path}/index.txt', 'r') as idx_file:
            for line in idx_file:
                md = json.loads(line.strip())
                if md['seed'] == seed and md['index'] == index and md['task'] == task and md['model'] == model and md['epochs'] == epochs and md['weight_decay'] == weight_decay:
                    raise ValueError(f'Model already exists as {md["id"]}')
    except FileNotFoundError:
        return


if __name__ == "__main__":
    print('Parsing args')
    args = parser.parse_args()
    random.seed(a=args.seed)

    assert_is_unique_model(args.path, args.seed, args.index, args.task, args.model, args.epochs, args.weight_decay)

    task = get_task_class(args.task)(args.seed)
    model = get_subject_model_class(args.model)
    net = model(task).to(DEVICE)

    train_subject_net(net, task.get_dataset(args.index), args.epochs, weight_decay=args.weight_decay)

    print("Evaluating models")
    loss = evaluate_subject_net(net, task.get_dataset(args.index, type=VAL))

    net_id = uuid.uuid4()
    md = vars(args)
    md.update(task.get_dataset(args.index).get_metadata())
    md["loss"] = loss

    if md["loss"] > 0.0001:
        print(md["loss"])

    md["id"] = str(net_id)
    md["time"] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    md["example_idx"] = args.index

    with open(f'{args.path}/index.txt', 'a') as md_file:
        md_file.write(json.dumps(md) + '\n')

    model_path = get_model_path(args.path, net_id) 
    torch.save(net.state_dict(), model_path)
