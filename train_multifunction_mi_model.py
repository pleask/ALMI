"""
Trains a network to recover both the function that the network implements, and
the coefficient.
"""
import argparse
import asyncio
from functools import cache
import json
import math
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import wandb

from train_subject_models import get_subject_net, FUNCTION_NAMES

os.environ["WANDB_SILENT"] = "true"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(a=0)

MI_CRITERION = nn.MSELoss()
SUBJECT_MODEL_PARAMETER_COUNT = 726
MI_MODEL_TRAIN_SPLIT_RATIO = 0.7
BATCH_SIZE=1024

def train_model(model, model_path, optimizer, epochs, train_dataloader, test_dataloader, test_dataset):
    model.train()
    for epoch in range(epochs):
        for _, (inputs, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = MI_CRITERION(outputs, targets)
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
            wandb.log({'epoch': epoch+1, 'loss': loss})
            torch.save(model.state_dict(), model_path)


def evaluate_model(model, test_dataloader):
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            print('-----')
            outputs = model(inputs)
            for target, output in zip(targets, outputs.squeeze()):
                print(FUNCTION_NAMES[torch.argmax(target[:5]).item()], target[-1].detach().item(), output[-1].detach().item())
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


class FeedForwardNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(SUBJECT_MODEL_PARAMETER_COUNT, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, len(FUNCTION_NAMES) + 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x


def get_matching_subject_models_names(subject_model_dir, max_loss, weight_decay):
    matching_subject_models_names = []

    index_file_path = f'{subject_model_dir}/index.txt'
    with open(index_file_path, 'r') as index_file:
        for line in index_file:
            line = line.strip()
            model_name, metadata_string = line.split(' ', maxsplit=1)
            metadata = json.loads(metadata_string)
            if metadata['weight_decay'] == weight_decay and metadata['loss'] <= max_loss:
                matching_subject_models_names.append(model_name)
        
    return matching_subject_models_names


@cache
def get_subject_model_metadata(subject_model_dir, subject_model_name):
    with open(f'{subject_model_dir}/{subject_model_name}_metadata.json') as f:
        return json.load(f)


class MultifunctionSubjectModelDataset(Dataset):
    """
    Dataset of subject models that match the weight decay specified.
    """
    def __init__(self, subject_model_dir, subject_model_names):
        self._subject_model_dir = subject_model_dir
        self.subject_model_names = subject_model_names

    def __len__(self):
        return len(self.subject_model_names)

    def __getitem__(self, idx):
        name = self.subject_model_names[idx]

        model = self._get_subject_model(name).to(DEVICE)
        x = torch.concat(
            [param.detach().reshape(-1) for _, param in model.named_parameters()]
        ).to(DEVICE)

        metadata = get_subject_model_metadata(self._subject_model_dir, name)
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
parser.add_argument("--model_type", type=str, help="Type of model to use.")
parser.add_argument("--model_path", type=str, help="Path to save this model")
parser.add_argument("--load_model", type=str, help="Path from which to load a model. Overrides model_path argument. Use in conjunction with --epochs=0 to just evaluate the model.")
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
    default=0.001
)

if __name__ == '__main__':
    os.environ["WAND_API_KEY"] = "dd685d7aa9b38a2289e5784a961b81e22fc4c735"

    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')

    wandb.init(config=args, project='bounding-mi', entity='patrickaaleask', reinit=True)

    print("Creating dataset", flush=True)
    all_matching_subject_models = get_matching_subject_models_names(args.subject_model_dir, args.max_loss, args.weight_decay)
    print(f"Found {len(all_matching_subject_models)}", flush=True)

    train_sample_count = int(len(all_matching_subject_models) * MI_MODEL_TRAIN_SPLIT_RATIO)
    print("Creating training dataset")
    train_dataset = MultifunctionSubjectModelDataset(args.subject_model_dir, all_matching_subject_models[:train_sample_count])
    print("Creating training dataloader")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Creating testing dataset")
    test_dataset = MultifunctionSubjectModelDataset(args.subject_model_dir, all_matching_subject_models[train_sample_count:])
    print("Creating testing dataloader")
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Creating model", flush=True)
    model_path = args.model_path

    model = FeedForwardNN().to(DEVICE)
    if args.model_type == 'transformer':
        model = Transformer(SUBJECT_MODEL_PARAMETER_COUNT, 6, num_heads=6, hidden_size=240).to(DEVICE)

    if args.load_model:
        print(f"Loading model {args.load_model}", flush=True)
        model.load_state_dict(torch.load(args.load_model))
        model_path = args.load_model

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    print("Training model", flush=True)
    train_model(model, model_path, optimizer, args.epochs, train_dataloader, test_dataloader, test_dataset)

    print("Prediction sample", flush=True)
    evaluate_model(model, test_dataloader)
