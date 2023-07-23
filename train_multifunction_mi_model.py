"""
Trains a network to recover both the function that the network implements, and
the coefficient.
"""
import argparse
from functools import cache
import json
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.prune import L1Unstructured
import wandb

from auto_mi.models import FeedForwardNN, SUBJECT_MODELS, FeedForwardNN2D
from auto_mi.tasks import TASKS

os.environ["WANDB_SILENT"] = "true"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MI_CRITERION = nn.MSELoss()
SUBJECT_MODEL_PARAMETER_COUNT = 726
MI_MODEL_TRAIN_SPLIT_RATIO = 0.7
BATCH_SIZE=1024

def train_model(model, model_path, optimizer, epochs, train_dataloader, test_dataloader, test_dataset):
    model.train()
    for _ in range(epochs):
        log = {}

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                outputs = model(inputs.to(DEVICE))
                loss = MI_CRITERION(outputs, targets.to(DEVICE))
                test_loss += loss.item() * inputs.size(0)
        avg_loss = test_loss / len(test_dataset)
        log = log | {'validation_loss': avg_loss}

        for (inputs, targets) in train_dataloader:
            log = log | {'loss': loss}
            optimizer.zero_grad()
            outputs = model(inputs.to(DEVICE))
            loss = MI_CRITERION(outputs, targets.to(DEVICE))
            loss.backward()
            optimizer.step()

            wandb.log(log)

        torch.save(model.state_dict(), model_path)

def evaluate_model(model, test_dataloader):
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            print('--- Sample output ---')
            outputs = model(inputs.to(DEVICE))
            for target, output in zip(targets, outputs.to(DEVICE)):
                print(target.detach(), output.detach())
            break


def get_matching_subject_models_names(subject_model_dir, task='SimpleFunctionRecoveryTask', max_loss=1., weight_decay=0.):
    matching_subject_models_names = []

    index_file_path = f'{subject_model_dir}/index.txt'
    with open(index_file_path, 'r') as index_file:
        for line in index_file:
            line = line.strip()
            metadata = json.loads(line)
            if metadata['task']['name'] != task:
                continue
            if weight_decay and metadata['trainer']['weight_decay'] != weight_decay:
                continue
            if max_loss and metadata['loss'] > max_loss:
                continue
            matching_subject_models_names.append(metadata['id'])
    return matching_subject_models_names


@cache
# TODO: use the index file for this
def get_subject_model_metadata(subject_model_dir, subject_model_name):
    with open(f'{subject_model_dir}/{subject_model_name}_metadata.json') as f:
        return json.load(f)

# This might run out of memory
@cache
def get_subject_model(net, subject_model_dir, subject_model_name, device='cuda'):
    if device=='cuda':
        net.load_state_dict(torch.load(f"{subject_model_dir}/{subject_model_name}.pickle"))
    else:
        net.load_state_dict(
            torch.load(f"{subject_model_dir}/{subject_model_name}.pickle", map_location=torch.device('cpu'))
        )
    return net


class MultifunctionSubjectModelDataset(Dataset):
    def __init__(self, subject_model_dir, subject_model_ids, prune_amount=0.):
        self._subject_model_dir = subject_model_dir
        self.subject_model_ids = subject_model_ids
        self._prune_amount = prune_amount

        self.metadata = self._index_metadata()

    def __len__(self):
        return len(self.subject_model_ids)

    def __getitem__(self, idx):
        name = self.subject_model_ids[idx]
        metadata = self.metadata[name]

        task = TASKS[metadata['task']['name']](metadata['seed'])
        example = task.get_dataset(metadata['index'])
        y = example.get_target()

        model = get_subject_model(SUBJECT_MODELS[metadata['model']['name']](task), self._subject_model_dir, name)

        # prune the weights of the model
        if self._prune_amount > 0.:
            pruner = L1Unstructured(self._prune_amount)
            for _, param in model.named_parameters():
                param.data = pruner.prune(param.data)

        x = torch.concat(
            [param.detach().reshape(-1) for _, param in model.named_parameters()]
        ).to(DEVICE)

        return x, y

    def _index_metadata(self):
        metadata = {}
        with open(f'{self._subject_model_dir}/index.txt', 'r') as f:
            for line in f:
                md = json.loads(line.strip())
                metadata[md['id']] = md
        return metadata

    @property
    def model_param_count(self):
        return self[0][0].shape[0]
    
    @property
    def output_shape(self):
        return self[0][1].shape


parser = argparse.ArgumentParser()
parser.add_argument("--repeat", help="Repeat number")
parser.add_argument("--subject_model_dir", help="Folder containing the subject models")
parser.add_argument("--model_type", type=str, help="Type of model to use.")
parser.add_argument("--model_path", type=str, help="Path to save this model")
parser.add_argument("--load_model", type=str, help="Path from which to load a model. Overrides model_path argument. Use in conjunction with --epochs=0 to just evaluate the model.")
parser.add_argument(
    "--epochs",
    type=int,
    help="Number of epochs for which to train.",
    default=1000,
)
parser.add_argument('--task', default='SimpleFunctionRecoveryTask', type=str, help='The MI task to learn.')
parser.add_argument(
    "--weight_decay",
    type=float,
    help="Weight decay for subject models",
    default=0.
)
parser.add_argument(
    "--max_loss",
    type=float,
    help="Max usable loss for the subject models",
    default=0.001
)
parser.add_argument(
    "--prune_amount",
    type=float,
    help="Amount by which to prune the subject models before training on them.",
    default=0.
)
parser.add_argument(
    "--layer_scale",
    type=float,
    help="Proportion by which to scale the size of the middle layers of the interpretability network.",
    default=1.
)

if __name__ == '__main__':
    os.environ["WAND_API_KEY"] = "dd685d7aa9b38a2289e5784a961b81e22fc4c735"

    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')

    random.seed(a=args.repeat)
    wandb.init(config=args, project='bounding-mi', entity='patrickaaleask', reinit=True)

    all_matching_subject_models = get_matching_subject_models_names(args.subject_model_dir, task=args.task, max_loss=args.max_loss, weight_decay=args.weight_decay)
    print(f"Found {len(all_matching_subject_models)}", flush=True)
    wandb.config['subject_model_count'] = len(all_matching_subject_models) 

    train_sample_count = int(len(all_matching_subject_models) * MI_MODEL_TRAIN_SPLIT_RATIO)
    train_dataset = MultifunctionSubjectModelDataset(args.subject_model_dir, all_matching_subject_models[:train_sample_count], prune_amount=args.prune_amount)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = MultifunctionSubjectModelDataset(args.subject_model_dir, all_matching_subject_models[train_sample_count:], prune_amount=args.prune_amount)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model_path = args.model_path

    task = TASKS[args.task]

    if len(train_dataset.output_shape) == 1:
        model = FeedForwardNN(train_dataset.model_param_count, train_dataset.output_shape[0], layer_scale=args.layer_scale).to(DEVICE)
    elif len(train_dataset.output_shape) == 2:
        model = FeedForwardNN2D(train_dataset.model_param_count, train_dataset.output_shape, layer_scale=args.layer_scale).to(DEVICE)

    if args.load_model:
        print(f"Loading model {args.load_model}", flush=True)
        model.load_state_dict(torch.load(args.load_model))
        model_path = args.load_model

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    print("Training model", flush=True)
    train_model(model, model_path, optimizer, args.epochs, train_dataloader, test_dataloader, test_dataset)

    print("Prediction sample", flush=True)
    evaluate_model(model, test_dataloader)
