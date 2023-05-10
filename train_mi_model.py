import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import json
import math
import os
import sys
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MI_MODEL_TRAIN_SPLIT_RATIO = 0.7
SUBJECT_MODEL_PARAMETER_COUNT = 726
MI_CRITERION = nn.MSELoss()
random.seed(a=0)

# TODO: commonise this with train_subject_models.py
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


def load_subject_model(model_path, model_idx):
    net = get_subject_net()
    # TODO: Commonise these paths across files
    net.load_state_dict(torch.load(f"{model_path}/{model_idx}.pickle"))
    with open(f"{model_path}/{model_idx}_metadata.json", "r") as f:
        metadata = json.load(f)
    return net, metadata


class SubjectModelDataset(Dataset):
    def __init__(self, start_idx, end_idx, model_path):
        """
        start_idx and end_idx are the lowest and highest subject model index that
        can be used in this dataset [start_idx, end_idx]. The models are indexed from
        1, so an example input here would be start_idx=1 end_idx=50.
        """
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.model_path = model_path

    def __len__(self):
        return self.end_idx - self.start_idx + 1

    # TODO: This is IO heavy, so can probably be parallelised (although actually seems quite fast sooo)
    def __getitem__(self, idx):
        net, metadata = load_subject_model(self.model_path, self.start_idx + idx)
        return torch.concat(
            [param.detach().reshape(-1) for _, param in net.named_parameters()]
        ), torch.tensor(metadata["parameter"], dtype=torch.float32)


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


parser = argparse.ArgumentParser(
    description="Trains MI models, ie. the models that recover the labeling function from the subject models."
)
parser.add_argument(
    "-s",
    "--subject_dir",
    type=str,
    required=True,
    help="Directory containing the subject models.",
)
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    help="Number of epochs for which to train. Will just evaluate the model if zero or unspecified. The loss seems to stop falling around 1k on addition, probably need to experiment for other functions.",
)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    help="Optionally load a model to evaluate or continue training.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    subject_dir = args.subject_dir
    epochs = args.epochs
    model_path = args.load_model

    print("Creating subject model dataset", flush=True)
    subject_model_count = len(os.listdir(subject_dir)) // 2
    train_split_count = int(MI_MODEL_TRAIN_SPLIT_RATIO * subject_model_count)
    train_dataset = SubjectModelDataset(1, train_split_count, subject_dir)
    test_dataset = SubjectModelDataset(
        train_split_count + 1, subject_model_count, subject_dir
    )
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=True)

    mi_model = Transformer(
        SUBJECT_MODEL_PARAMETER_COUNT, 1, num_heads=6, hidden_size=240
    ).to(DEVICE)
    mi_optimizer = torch.optim.Adam(mi_model.parameters(), lr=0.00001)

    if model_path:
        print("loading existing model", flush=True)
        mi_model.load_state_dict(torch.load(model_path))
    else:
        model_path = 'mi_model.pickle'

    print("training mi transformer", flush=True)
    mi_model.train()
    outputs = None
    targets = None
    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(train_dataloader):
            targets = targets.to(DEVICE)
            mi_optimizer.zero_grad()
            outputs = mi_model(inputs)
            loss = MI_CRITERION(outputs, targets.unsqueeze(1))
            loss.backward()
            mi_optimizer.step()

        if epoch % 100 == 0:
            print(f"mi model epoch {epoch} of {epochs}", flush=True)
            mi_model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    targets = targets.to(DEVICE)
                    outputs = mi_model(inputs)
                    loss = MI_CRITERION(outputs, targets.unsqueeze(1))
                    test_loss += loss.item() * inputs.size(0)
            avg_loss = test_loss / len(test_dataset)
            print(f"\nTest Loss: {avg_loss:.4f}", flush=True)
    print("Last batch as example", flush=True)
    print(torch.cat((outputs, targets.unsqueeze(1)), dim=1), flush=True)
    torch.save(mi_model.state_dict(), model_path)
