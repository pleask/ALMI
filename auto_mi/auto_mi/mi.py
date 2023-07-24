from functools import cache
import json

import torch
from torch.nn.utils.prune import L1Unstructured
from torch.utils.data import Dataset
import wandb

from auto_mi.tasks import TASKS
from auto_mi.models import SUBJECT_MODELS


def train_model(model, model_path, optimizer, epochs, train_dataloader, test_dataloader, test_dataset, criterion, device='cpu'):
    model.train()
    for _ in range(epochs):
        log = {}

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                outputs = model(inputs.to(device))
                loss = criterion(outputs, targets.to(device))
                test_loss += loss.item() * inputs.size(0)
        avg_loss = test_loss / len(test_dataset)
        log = log | {'validation_loss': avg_loss}

        for (inputs, targets) in train_dataloader:
            log = log | {'loss': loss}
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            loss.backward()
            optimizer.step()

            wandb.log(log)

        torch.save(model.state_dict(), model_path)


def evaluate_model(model, test_dataloader, device='cpu'):
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            print('--- Sample output ---')
            outputs = model(inputs.to(device))
            for target, output in zip(targets, outputs.to(device)):
                print(target.detach(), output.detach())
            break


def get_matching_subject_models_names(subject_model_dir, task='SimpleFunctionRecoveryTask', max_loss=1., weight_decay=0., prune_amount=0.):
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
            if prune_amount and metadata['trainer']['prune_amount'] != prune_amount:
                continue
            if max_loss and metadata['loss'] > max_loss:
                continue
            matching_subject_models_names.append(metadata['id'])
    return matching_subject_models_names


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
    def __init__(self, subject_model_dir, subject_model_ids, device='cpu'):
        self._subject_model_dir = subject_model_dir
        self.subject_model_ids = subject_model_ids
        self.device = device

        self.metadata = self._index_metadata()

    def __len__(self):
        return len(self.subject_model_ids)

    def __getitem__(self, idx):
        name = self.subject_model_ids[idx]
        metadata = self.metadata[name]

        task = TASKS[metadata['task']['name']](metadata['task']['seed'])
        example = task.get_dataset(metadata['index'])
        y = example.get_target()

        model = get_subject_model(SUBJECT_MODELS[metadata['model']['name']](task), self._subject_model_dir, name)

        x = torch.concat(
            [param.detach().reshape(-1) for _, param in model.named_parameters()]
        ).to(self.device)

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