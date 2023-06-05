"""
Trains a network to recover both the function that the network implements, and
the coefficient.
"""
import argparse
from functools import cache
import json
import math
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_subject_models import get_subject_net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(a=0)

# TODO: Commonise this with the subject models script
FUNCTION_NAMES = ['addition', 'multiplication', 'sigmoid', 'exponent', 'min']
MI_CRITERION = nn.MSELoss()
SUBJECT_MODEL_PARAMETER_COUNT = 726
MI_MODEL_TRAIN_SPLIT_RATIO = 0.7

def train_model(model, model_path, optimizer, epochs, train_dataloader, test_dataloader, test_dataset):
    model.train()
    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = MI_CRITERION(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()

        # after every 10 epochs, evaluate and save the model
        if epoch % 10 == 9:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    targets = targets.to(DEVICE)
                    outputs = model(inputs)
                    loss = MI_CRITERION(outputs, targets.unsqueeze(1))
                    test_loss += loss.item() * inputs.size(0)
            avg_loss = test_loss / len(test_dataset)
            print(f"Epoch {epoch+1} of {epochs}. Test Loss: {avg_loss:.4f}", flush=True)
            torch.save(model.state_dict(), model_path)


def evaluate_model(model, test_dataloader, test_dataset):
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            outputs = model(inputs)
            [print(test_dataset.get_dataset_index(i), t.detach().cpu().item(), o.detach().cpu().item()) for i, (t, o) in enumerate(zip(targets, outputs.squeeze()))]
            break

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


class Transformer(nn.Module):
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

def get_matching_subject_models_names(subject_model_dir, max_loss, weight_decay):
    all_subject_model_filenames = os.listdir(subject_model_dir)
    matching_names = []
    for filename in all_subject_model_filenames:
        if not filename.endswith('.pickle'):
            continue
        subject_model_name = filename.removesuffix('.pickle')
        metadata = get_subject_model_metadata(subject_model_name)
        if metadata['loss'] > max_loss:
            pass
        if metadata['weight_decay'] != weight_decay:
            pass
        matching_names.append(subject_model_name)
    return matching_names


@cache
def get_subject_model_metadata(self, subject_model_name):
    with open(f'{self._subject_model_dir}/{subject_model_name}_metadata.json') as f:
        return json.load(f)


class MultifunctionSubjectModelDataset(Dataset):
    """
    Dataset of subject models that match the weight decay specified.
    """
    def __init__(self, subject_model_dir, subject_model_names):
        self._subject_model_dir = subject_model_dir
        self._subject_model_names = subject_model_names

    def __len__(self):
        return len(self._subject_models)

    def __getitem__(self, idx):
        name = self._subject_model_names[idx]

        model = self._get_subject_model(name)
        x = torch.concat(
            [param.detach().reshape(-1) for _, param in model.named_parameters()]
        )

        metadata = self._get_subject_model_metadata(name)
        fn_name = metadata['fn_name']
        parameter = metadata['parameter']
        one_hot = [0.] * len(FUNCTION_NAMES)
        one_hot[FUNCTION_NAMES.index(fn_name)] = 1.
        one_hot.append(parameter)
        y = torch.tensor(one_hot).to(DEVICE)

        return x, y

    # This might run out of memory
    @cache
    def _get_subject_model(self, subject_model_name):
        net = get_subject_net()
        net.load_state_dict(torch.load(f"{self._subject_model_dir}/{subject_model_name}.pickle"))
        return net


parser = argparse.ArgumentParser()
parser.add_argument("--subject_model_dir", help="Folder containing the subject models")
parser.add_argument("--model_path", type=str, help="Path to save this model")
parser.add_argument(
    "--epochs",
    type=int,
    help="Number of epochs for which to train.",
    default=1000,
)
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
    default=0.0001
)

if __name__ == '__main__':
    args = parser.parse_args()

    print("Creating dataset", flush=True)
    all_matching_subject_models = get_matching_subject_models_names(args.subject_model_dir, args.max_loss, args.weight_decay)
    train_sample_count = int(all_matching_subject_models * MI_MODEL_TRAIN_SPLIT_RATIO)
    train_dataset = MultifunctionSubjectModelDataset(args.subject_model_dir, all_matching_subject_models[:train_sample_count])
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_dataset = MultifunctionSubjectModelDataset(args.subject_model_dir, all_matching_subject_models[train_sample_count:])
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=True)

    print("Creating model", flush=True)
    model = Transformer(SUBJECT_MODEL_PARAMETER_COUNT, 6, num_heads=6, hidden_size=240)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    print("Training model", flush=True)
    train_model(model, args.model_path, optimizer, args.epochs, train_dataloader, test_dataloader, test_dataset)

    print("Prediction sample", flush=True)
    evaluate_model(model, test_dataloader, test_dataset)
