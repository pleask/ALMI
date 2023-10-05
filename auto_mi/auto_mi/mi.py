import json
import math
import os
import tarfile
import tempfile
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm

from auto_mi.tasks import TASKS, MI
from auto_mi.models import SUBJECT_MODELS
from auto_mi.base import MetadataBase
from auto_mi.tasks import SimpleFunctionRecoveryTask

TRAIN_RATIO = 0.
INTERPRETABILITY_BATCH_SIZE = 2**7

# TODO: Add wandb logging of the trained models
def train_interpretability_model(model, task, subject_model_path, validation_subject_models, trainer, reuse_count=1000):
    device = model.device

    # Train the interpretability model on all the subject models that are not
    # going to be used for validation.
    training_model_names, _ = get_matching_subject_models_names(subject_model_path, trainer=trainer, task=task, exclude=validation_subject_models)
    # Use a subsample of the existing models at each RL step rather than constantly retraining the model on everything.
    training_model_names = random.sample(training_model_names, min([reuse_count, len(training_model_names)]))
    print(f'Using {len(training_model_names)} subject models')

    wandb.log({'subject_model_count': training_model_names})
    train_dataset = MultifunctionSubjectModelDataset(subject_model_path, training_model_names)
    train_dataloader = DataLoader(train_dataset, batch_size=INTERPRETABILITY_BATCH_SIZE, shuffle=True, num_workers=0)

    # The performance of the interpretability model is evaluated on the
    # validation subject models, which were all created by the same trainer in
    # this step.
    eval_dataset = MultifunctionSubjectModelDataset(subject_model_path, validation_subject_models)
    eval_dataloader = DataLoader(eval_dataset, batch_size=INTERPRETABILITY_BATCH_SIZE, shuffle=True, num_workers=0)

    # TODO: Take these as a parameter as will vary by task and model.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.MSELoss()
    epochs = 30

    model.train()
    for epoch in tqdm(range(epochs), desc='Interpretability model epochs'):
        train_loss = 0.
        for i, (inputs, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs.to(device, non_blocking=True))
            loss = criterion(outputs, targets.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
            train_loss += loss

    model.eval()
    with torch.no_grad():
        eval_loss = 0.
        for inputs, targets in eval_dataloader:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            eval_loss += loss.item()
    
    return eval_loss / len(eval_dataloader)


def get_matching_subject_models_names(subject_model_dir, trainer, task=None, exclude=None):
    if exclude is None:
        exclude = []
    if task is None:
        task = SimpleFunctionRecoveryTask
        
    matching_subject_models_names = []
    losses = []
    
    # Check if subject_model_dir is a tar archive
    is_tar_archive = tarfile.is_tarfile(subject_model_dir)

    if is_tar_archive:
        with tarfile.open(subject_model_dir, 'r') as archive:
            index_member = archive.getmember('index.txt')
            with archive.extractfile(index_member) as index_file:
                _process_index_file(index_file, subject_model_dir, task, trainer, matching_subject_models_names, losses, exclude, is_tar_archive)
    else:
        index_file_path = os.path.join(subject_model_dir, 'index.txt')
        with open(index_file_path, 'r') as index_file:
            _process_index_file(index_file, subject_model_dir, task, trainer, matching_subject_models_names, losses, exclude, is_tar_archive)
    
    return matching_subject_models_names, sum(losses) / len(losses) if losses else 0

def _process_index_file(index_file, subject_model_dir, task, trainer, matching_subject_models_names, losses, exclude, is_tar_archive):
    for line in index_file:
        line = line.strip()
        metadata = json.loads(line)

        if metadata['task']['name'] != type(task).__name__:
            continue
        model_filename = f'{metadata["id"]}.pickle'
        model_exists = _check_model_exists(subject_model_dir, model_filename, is_tar_archive)
        if not model_exists:
            print(f'Model {metadata["id"]} does not exist')
            continue

        trainer_metadata = trainer.get_metadata()
        if not all(metadata['trainer'][key] == trainer_metadata[key] for key in ['name', 'weight_decay', 'lr', 'prune_amount']):
            continue

        if metadata['id'] in exclude:
            continue

        matching_subject_models_names.append(metadata['id'])
        losses.append(metadata['id'])

def _check_model_exists(subject_model_dir, model_filename, is_tar_archive):
    if is_tar_archive:
        with tarfile.open(subject_model_dir, 'r') as archive:
            try:
                archive.getmember(model_filename)
                return True
            except KeyError:  # file not found in archive
                return False
    else:
        return os.path.exists(os.path.join(subject_model_dir, model_filename))


def get_subject_model(net, subject_model_dir, subject_model_name, device='cuda'):
    model_filename = f"{subject_model_name}.pickle"
    
    def load_model(file_path):
        """Loads a model's state dict from a file."""
        try:
            load_args = (file_path,) if device == 'cuda' else (file_path, {'map_location': torch.device('cpu')})
            net.load_state_dict(torch.load(*load_args))
        except (OSError, EOFError, RuntimeError) as e:
            raise Exception(f'Failed on {subject_model_name}: {e}') from e
    
    # Check if the subject_model_dir is a tar archive
    if tarfile.is_tarfile(subject_model_dir):
        # Use a temporary directory to extract our model file to.
        with tempfile.TemporaryDirectory() as tmp_dir:
            with tarfile.open(subject_model_dir, 'r') as tar:
                try:
                    # Extract the specific model file from the tar archive.
                    tar.extract(model_filename, path=tmp_dir)
                except KeyError:
                    raise Exception(f'{model_filename} not found in the tar archive.')
                else:
                    # Load the model from the extracted file.
                    load_model(os.path.join(tmp_dir, model_filename))
    else:  # If it's not a tar archive, proceed as in the original function
        model_filepath = os.path.join(subject_model_dir, model_filename)
        load_model(model_filepath)
    
    return net


class MultifunctionSubjectModelDataset(Dataset):
    def __init__(self, subject_model_dir, subject_model_ids, device='cpu'):
        self._subject_model_dir = subject_model_dir
        self.subject_model_ids = subject_model_ids
        self.device = device

        self.metadata = self._index_metadata()
        self.cache = {}

    def __len__(self):
        return len(self.subject_model_ids)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx][0], self.cache[idx][1]
        name = self.subject_model_ids[idx]
        metadata = self.metadata[name]

        task = TASKS[metadata['task']['name']](**metadata['task'])
        example = task.get_dataset(metadata['index'], purpose=MI)
        y = example.get_target()

        model = get_subject_model(SUBJECT_MODELS[metadata['model']['name']](task), self._subject_model_dir, name)

        x = torch.concat(
            [param.detach().reshape(-1) for _, param in model.named_parameters()]
        )

        del model

        self.cache[idx] = (x, y)

        return x, y

    def _index_metadata(self):
        def process_lines(file):
            """Process lines from a file-like object and build metadata dictionary."""
            local_metadata = {}
            for line in file:
                md = json.loads(line.strip().decode('utf-8') if isinstance(line, bytes) else line.strip())
                local_metadata[md['id']] = md
            return local_metadata
        
        metadata = {}
        # Check if the subject_model_dir is a tar archive
        if tarfile.is_tarfile(self._subject_model_dir):
            # Open the tar file
            with tarfile.open(self._subject_model_dir, 'r') as tar:
                try:
                    with tar.extractfile('index.txt') as index_file:
                        metadata = process_lines(index_file)
                except KeyError:
                    print("index.txt not found in the tar archive.")
        # If it's not a tar archive, then proceed as before
        else:
            index_path = os.path.join(self._subject_model_dir, 'index.txt')
            if os.path.exists(index_path):
                with open(index_path, 'r') as f:
                    metadata = process_lines(f)
            else:
                print("index.txt not found in the directory.")
        
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
        in_size,
        out_shape,
        num_layers=6,
        hidden_size=256,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        
        self.out_shape = out_shape
        output_size = torch.zeros(out_shape).view(-1).shape[0]

        self.embedding = nn.Linear(in_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = self.output_layer(x.mean(dim=0))  # use mean pooling to obtain a single output value
        x = x.view(-1, *self.out_shape)
        return self.softmax(x)

    @property
    def device(self):
        return self.output_layer.weight.device


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
