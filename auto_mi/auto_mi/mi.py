import json
import math
from auto_mi.base import MetadataBase
from auto_mi.tasks import SimpleFunctionRecoveryTask

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import wandb

from auto_mi.tasks import TASKS, MI
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
            print('--- Sample output ---', flush=True)
            outputs = model(inputs.to(device))
            for target, output in zip(targets, outputs.to(device)):
                print(target.detach(), output.detach(), flush=True)
            break


def get_matching_subject_models_names(subject_model_dir, task=SimpleFunctionRecoveryTask, max_loss=1., weight_decay=0., prune_amount=0.):
    matching_subject_models_names = []

    index_file_path = f'{subject_model_dir}/index.txt'
    with open(index_file_path, 'r') as index_file:
        for line in index_file:
            line = line.strip()
            metadata = json.loads(line)
            if metadata['task']['name'] != task.__name__:
                continue
            if weight_decay and metadata['trainer']['weight_decay'] != weight_decay:
                continue
            if prune_amount and metadata['trainer']['prune_amount'] != prune_amount:
                continue
            if max_loss and metadata['loss'] > max_loss:
                continue
            matching_subject_models_names.append(metadata['id'])
    return matching_subject_models_names


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

        task = TASKS[metadata['task']['name']](seed=metadata['task']['seed'])
        example = task.get_dataset(metadata['index'], purpose=MI)
        y = example.get_target()

        model = get_subject_model(SUBJECT_MODELS[metadata['model']['name']](task), self._subject_model_dir, name)

        x = torch.concat(
            [param.detach().reshape(-1) for _, param in model.named_parameters()]
        ).to(self.device)

        del model

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


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_seq_len=5000, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, hidden_size)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module, MetadataBase):
    def __init__(
        self,
        input_size,
        output_size,
        num_layers=6,
        hidden_size=256,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = self.output_layer(
            x.mean(dim=0)
        )  # use mean pooling to obtain a single output value
        return x


class FeedForwardNN(nn.Module, MetadataBase):
    def __init__(self, in_size, out_size, layer_scale=1):
        super().__init__()
        self.fc1 = nn.Linear(in_size, int(128*layer_scale))
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(int(128*layer_scale), int(64*layer_scale))
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(int(64*layer_scale), out_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        function_encoding = self.softmax(x[:, :-1])
        x = torch.cat([function_encoding, x[:, -1:]], dim=-1)
        return x


class SimpleFunctionRecoveryModel(nn.Module, MetadataBase):
    def __init__(self, in_size, out_shape, layer_scale=1):
        super().__init__()
        self.out_shape = out_shape
        out_size = torch.zeros(out_shape).view(-1).shape[0]
        self.fc1 = nn.Linear(in_size, int(128*layer_scale))
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(int(128*layer_scale), int(64*layer_scale))
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(int(64*layer_scale), out_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = x.view(-1, *self.out_shape)
        function_encoding = self.softmax(x[:, :, :-1])
        x = torch.cat([function_encoding, x[:, :, -1:]], dim=-1)
        return x


class IntegerGroupFunctionRecoveryModel(nn.Module, MetadataBase):
    def __init__(self, in_size, out_shape, layer_scale=5):
        super().__init__()
        self.out_shape = out_shape
        out_size = torch.zeros(out_shape).view(-1).shape[0]
        self.fc1 = nn.Linear(in_size, int(64*layer_scale))
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(int(64*layer_scale), int(32*layer_scale))
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(int(32*layer_scale), out_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = x.view(-1, *self.out_shape)
        return self.softmax(x)
