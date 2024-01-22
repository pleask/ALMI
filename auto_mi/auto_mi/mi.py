"""
Methods for training and evaluating interpretability models.
"""
import os
import math
from auto_mi.subject_models import get_matching_subject_models_names

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm

from auto_mi.tasks import MI
from auto_mi.base import MetadataBase

VAL_RATIO = 0.2
TRAIN_RATIO = 0.8
INTERPRETABILITY_BATCH_SIZE = 2**7
CRITERION = nn.BCEWithLogitsLoss()


# TODO: subject_model can go into the IO class rather than be passed in here
def train_mi_model(
    run,
    interpretability_model,
    interpretability_model_io,
    subject_model,
    subject_model_io,
    trainer,
    task,
    batch_size=2**7,
    epochs=1000,
    device="cuda",
    lr=1e-5,
    subject_model_count=-1,
    frozen_layers=None,
    num_classes=-1,
    subject_model_example_count=-1,
):
    """
    Trains an interpretability transformer model on the specified subject models.
    """
    all_subject_models, _ = get_matching_subject_models_names(
        subject_model_io,
        trainer,
        task=task,
        frozen_layers=frozen_layers,
        num_classes=num_classes,
        subject_model_example_count=subject_model_example_count,
    )
    if subject_model_count > 0:
        all_subject_models = all_subject_models[:subject_model_count]
    validation_models, train_models = (
        all_subject_models[: int(VAL_RATIO * len(all_subject_models))],
        all_subject_models[int(VAL_RATIO * len(all_subject_models)) :],
    )

    total_model_count = len(validation_models) + len(train_models)
    if total_model_count == 0:
        raise ValueError("No subject models found")
    wandb.log({"subject_model_count": total_model_count})
    print(f"Using {total_model_count} subject models")

    train_dataset = MultifunctionSubjectModelDataset(
        subject_model_io,
        train_models,
        task,
        subject_model,
        normalise=True,
        frozen_layers=frozen_layers,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    # Split the validation models into groups with the same frozen layers for better logging to wandb.
    frozen_layers_to_ids = {}
    # the dataset metadata has all the models in the directory, regardless of
    # whether they are used in the training dataset
    for id in validation_models:
        try:
            frozen_layers = tuple(train_dataset.metadata[id]["model"]["frozen"])
            if frozen_layers in frozen_layers_to_ids:
                frozen_layers_to_ids[frozen_layers].append(id)
            else:
                frozen_layers_to_ids[frozen_layers] = [id]
        except KeyError:
            pass
    validation_dataloaders = {}
    for frozen_layers, ids in frozen_layers_to_ids.items():
        validation_dataloaders[frozen_layers] = DataLoader(
            MultifunctionSubjectModelDataset(
                subject_model_io,
                ids,
                task,
                subject_model,
                normalise=True,
                frozen_layers=frozen_layers,
            ),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

    # Log a histogram of the subject model losses
    subject_model_losses = [
        train_dataset.metadata[id]["loss"] for id in train_models + validation_models
    ]
    data = [[s] for s in subject_model_losses]
    table = wandb.Table(data=data, columns=["losses"])
    wandb.log(
        {
            "my_histogram": wandb.plot.histogram(
                table, "losses", title="Subject model losses"
            )
        }
    )

    optimizer = torch.optim.Adam(
        interpretability_model.parameters(), lr=lr, weight_decay=0.001
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=20, factor=0.1, verbose=True
    )

    device_cap = torch.cuda.get_device_capability()
    if device_cap in ((7, 0), (8, 0), (9, 0)):
        interpretability_model = torch.compile(interpretability_model)
        torch.backends.cuda.matmul.allow_tf32 = True

    for epoch in tqdm(range(epochs), desc="Interpretability model epochs"):
        interpretability_model.train()
        for i, (inputs, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = interpretability_model(inputs.to(device, non_blocking=True))
            loss = 0
            for i in range(outputs.shape[1]):
                loss += CRITERION(outputs[:, i], targets[:, i].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(interpretability_model.parameters(), 0.5)
            optimizer.step()
            wandb.log({"train_loss": loss})

        scheduler.step(loss)
        eval_loss, accuracy = _evaluate(
            interpretability_model, validation_dataloaders, device=device
        )
        wandb.log(
            {
                **{
                    f"validation_loss_{frozen_layers}": loss
                    for frozen_layers, loss in eval_loss.items()
                },
                **{
                    f"validation_accuracy_{frozen_layers}": acc
                    for frozen_layers, acc in accuracy.items()
                },
            }
        )

        interpretability_model_io.write_model(run.id, interpretability_model)


def _evaluate(interpretability_model, validation_dataloaders, device="cuda"):
    accuracies = {}
    eval_losses = {}

    interpretability_model.eval()
    with torch.no_grad():
        for frozen_layers, validation_dataloader in validation_dataloaders.items():
            eval_loss = 0.0
            accuracy = 0.0
            for inputs, targets in validation_dataloader:
                outputs = interpretability_model(inputs.to(device))
                loss = CRITERION(outputs, targets.to(device))
                eval_loss += loss.item()
                predicted_classes = torch.argmax(outputs, dim=-1)
                target_classes = torch.argmax(targets, dim=-1)
                accuracy += torch.sum(
                    torch.all(
                        predicted_classes.detach().cpu() == target_classes.cpu(), dim=1
                    )
                ).item() / len(predicted_classes)

            eval_loss /= len(validation_dataloader)
            eval_losses[frozen_layers] = eval_loss

            accuracy /= len(validation_dataloader)
            accuracies[frozen_layers] = accuracy

    return eval_losses, accuracies


def evaluate_interpretability_model(
    interpretability_model_id,
    interpretability_model,
    interpretability_model_io,
    subject_model,
    subject_model_io,
    task,
    trainer,
    subject_model_count=100,
    batch_size=2**5,
    device="cuda",
):
    """
    Evaluates an interpretability model on the specified subject models.

    validate_on_non_frozen: If set to true, validates only on non-frozen models.
    """
    raise NotImplementedError("Method removed in favour of wandb logging.")


class MultifunctionSubjectModelDataset(Dataset):
    def __init__(
        self,
        model_loader,
        subject_model_ids,
        task,
        subject_model,
        normalise=False,
        device="cpu",
        frozen_layers=None,
    ):
        self._model_loader = model_loader
        self.subject_model_ids = subject_model_ids
        self.device = device
        self.task = task
        self.subject_model = subject_model
        self.frozen_layers = frozen_layers

        self.metadata = self._index_metadata()
        self._data = [None for _ in self.subject_model_ids]

        # Initially set normalise to false so we don't attempt to retrieve normalised data for normalisation
        self._normalise = False
        if normalise:
            samples = [self[i][0] for i in range(len(subject_model_ids) // 1)]
            params = torch.stack(samples).view(-1)
            self._std, self._mean = torch.std_mean(params, dim=-1)
            self._normalise = True

    def __len__(self):
        return len(self.subject_model_ids)

    def __getitem__(self, idx):
        if self._data[idx]:
            return self._data[idx]
        name = self.subject_model_ids[idx]
        metadata = self.metadata[name]

        example = self.task.get_dataset(metadata["index"], type=MI)
        # TODO: Something is broken with the datasets so here's a workaround but fix asap
        example._permutation_map = metadata["example"]["permutation_map"]
        y = example.get_target()

        model = self._model_loader.get_model(self.subject_model(self.task), name)

        frozen_param_count = 0
        if self.frozen_layers is not None:
            # count the number of parameters in the frozen layers
            for i, param in enumerate(model.parameters()):
                if i // 2 in self.frozen_layers:
                    frozen_param_count += param.numel()

        x = torch.concat(
            [param.detach().reshape(-1) for _, param in model.named_parameters()]
        )

        x = x[frozen_param_count:]

        if self._normalise:
            x = (x - self._mean) / self._std

        self._data[idx] = (x, y)

        return x, y

    def _index_metadata(self):
        d = {}
        for md in self._model_loader.get_metadata():
            d[md["id"]] = md
        return d

    @property
    def model_param_count(self):
        return self[0][0].shape[0]

    @property
    def output_shape(self):
        return self[0][1].shape


class PositionalEncoding(nn.Module):
    def __init__(self, encoding_length=4, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.length = encoding_length
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, encoding_length, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / encoding_length)
        )

        pe = torch.zeros(1, max_len, encoding_length)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe[:, : x.size(1), :].expand(x.shape[0], -1, -1)
        pe = pe.expand(x.shape[0], -1, -1)
        return x + pe


class Transformer(nn.Module, MetadataBase):
    def __init__(
        self,
        subject_model_parameter_count,
        out_shape,
        num_layers=6,
        num_heads=1,
        positional_encoding_size=4096,
    ):
        super().__init__()
        self.out_shape = out_shape

        output_size = torch.zeros(out_shape).view(-1).shape[0]
        self.positional_encoding = PositionalEncoding(
            positional_encoding_size, subject_model_parameter_count
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.positional_encoding.length,
            nhead=num_heads,
            dim_feedforward=2048,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.positional_encoding.length),
        )

        # Linear layer for classification
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.positional_encoding.length, output_size)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.expand(x.shape[0], x.shape[1], self.positional_encoding.length)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)

        x = x.transpose(1, 2)
        x = self.global_avg_pooling(x).squeeze(2)
        output = self.fc(x)
        output = output.view(-1, *self.out_shape)
        return output


class EmbeddingTransformer(nn.Module, MetadataBase):
    """
    Chunks the input and embeds each chunk separately before passing it through
    the transformer.
    """
    def __init__(
        self,
        subject_model_parameter_count,
        out_shape,
        num_layers=6,
        num_heads=1,
        positional_encoding_size=4096,
        chunk_size = 64,
    ):
        super().__init__()
        self.out_shape = out_shape

        output_size = torch.zeros(out_shape).view(-1).shape[0]

        self.positional_encoding = PositionalEncoding(
            positional_encoding_size, subject_model_parameter_count
        )

        self.embedding = nn.Linear(chunk_size, positional_encoding_size)
        self._chunk_size = chunk_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.positional_encoding.length,
            nhead=num_heads,
            dim_feedforward=2048,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.positional_encoding.length),
        )

        # Linear layer for classification
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.positional_encoding.length, output_size)
    
    def _chunk_input(self, x):
        _, seq_len = x.size()
        num_chunks = (seq_len - 1) // self._chunk_size + 1
        chunks = []
        for i in range(num_chunks):
            start = i * self._chunk_size
            end = min((i + 1) * self._chunk_size, seq_len)
            chunk = x[:, start:end]
            if chunk.size(1) < self._chunk_size:
                padding = torch.zeros((chunk.size(0), self._chunk_size - chunk.size(1))).to(chunk.device)
                chunk = torch.cat((chunk, padding), dim=1)
            chunks.append(chunk)
        stacked_chunks = torch.stack(chunks, dim=1)
        return stacked_chunks

    def forward(self, x):
        x = self._chunk_input(x)
        x = self.embedding(x) * math.sqrt(self.positional_encoding.length)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.global_avg_pooling(x).squeeze(2)
        output = self.fc(x)
        output = output.view(-1, *self.out_shape)
        return output


class FreezableClassifier:
    def __init__(self, file):
        script_dir = os.path.dirname(os.path.abspath(file))
        base_model_path = os.path.join(script_dir, "base_model.pickle")
        checkpoint = torch.load(base_model_path)
        self.load_state_dict(checkpoint)
