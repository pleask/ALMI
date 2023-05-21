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
        self.model_path = model_path
        self.start_idx = start_idx
        self.data = self._load_data(model_path, start_idx, end_idx)

    @staticmethod
    def _load_data(model_path, start_idx, end_idx):
        data = []
        for i in range(start_idx, end_idx):
            net, metadata = load_subject_model(model_path, i)
            # filter out models with a high loss
            if metadata['loss'] > 0.00001:
                continue
            data.append((i, net, metadata))
        return data 

    def get_dataset_index(self, i):
        """
        Retrieve the index for the whole dataset from the index for this subset.
        """
        return self.start_idx + self.data[i][0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, net, metadata = self.data[idx]
        x = torch.concat(
            [param.detach().reshape(-1) for _, param in net.named_parameters()]
        )
        y = torch.tensor(metadata["parameter"], dtype=torch.float32, device=DEVICE)
        return x, y


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
    "-d",
    "--dir",
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
# TODO: replace this with a check to see if there's already a model there, and a flag to overwrite
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    help="Optionally load a model to evaluate or continue training.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    epochs = args.epochs
    dir = args.dir
    model_path = args.model_path
    subject_dir = f'{dir}/subject_models'

    print("CREATING DATASET", flush=True)
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
        print("LOADING MODEL", flush=True)
        mi_model.load_state_dict(torch.load(model_path))
    else:
        model_path = f'{dir}/mi_model.pickle'

    print("TRAINING", flush=True)
    mi_model.train()
    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(train_dataloader):
            mi_optimizer.zero_grad()
            outputs = mi_model(inputs)
            loss = MI_CRITERION(outputs, targets.unsqueeze(1))
            loss.backward()
            mi_optimizer.step()

        # after every 10 epochs, evaluate and save the model
        if epoch % 10 == 9:
            mi_model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    targets = targets.to(DEVICE)
                    outputs = mi_model(inputs)
                    loss = MI_CRITERION(outputs, targets.unsqueeze(1))
                    test_loss += loss.item() * inputs.size(0)
            avg_loss = test_loss / len(test_dataset)
            print(f"Epoch {epoch+1} of {epochs}. Test Loss: {avg_loss:.4f}", flush=True)
            torch.save(mi_model.state_dict(), model_path)

    print("PREDICTION SAMPLE", flush=True)
    mi_model.eval()
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            outputs = mi_model(inputs)
            [print(test_dataset.get_dataset_index(i), t.detach().cpu().item(), o.detach().cpu().item()) for i, (t, o) in enumerate(zip(targets, outputs.squeeze()))]
            break
