"""
Methods for training and evaluating interpretability models.
"""
import os
import random

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
    validate_on_non_frozen=True,
):
    """
    Trains an interpretability transformer model on the specified subject models.

    validate_on_non_frozen: If set to true, validates only on non-frozen models.
    """
    if validate_on_non_frozen:
        frozen_subject_models, _ = get_matching_subject_models_names(
            subject_model_io, trainer, task=task, frozen=FROZEN_ONLY
        )
        non_frozen_subject_models, _ = get_matching_subject_models_names(
            subject_model_io, trainer, task=task, frozen=NON_FROZEN_ONLY
        )
        train_non_frozen = non_frozen_subject_models[
            int(VAL_RATIO * len(non_frozen_subject_models)) :
        ]
        validation_models = non_frozen_subject_models[
            : int(VAL_RATIO * len(non_frozen_subject_models))
        ]
        train_models = frozen_subject_models + train_non_frozen
    else:
        all_subject_models, _ = get_matching_subject_models_names(
            subject_model_io, trainer, task=task
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
        subject_model_io, train_models, task, subject_model, normalise=True
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    validation_dataset = MultifunctionSubjectModelDataset(
        subject_model_io, validation_models, task, subject_model, normalise=True
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    optimizer = torch.optim.Adam(
        interpretability_model.parameters(), lr=lr, weight_decay=0.001
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.1, verbose=True
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
            interpretability_model, validation_dataloader, device=device
        )
        wandb.log({"validation_loss": eval_loss, "validation_accuracy": accuracy})

        # TODO: Save to a more usable filename
        interpretability_model_io.write_model(
            f"{trainer.get_metadata()}", interpretability_model
        )


def _evaluate(interpretability_model, validation_dataloader, device="cuda"):
    interpretability_model.eval()
    with torch.no_grad():
        eval_loss = 0.0
        for inputs, targets in validation_dataloader:
            outputs = interpretability_model(inputs.to(device))
            loss = CRITERION(outputs, targets.to(device))
            eval_loss += loss.item()

        eval_loss /= len(validation_dataloader)

        predicted_classes = torch.argmax(outputs, dim=-1)
        target_classes = torch.argmax(targets, dim=-1)
        accuracy = torch.sum(
            torch.all(predicted_classes.detach().cpu() == target_classes.cpu(), dim=1)
        ).item() / len(predicted_classes)
    return eval_loss, accuracy


def evaluate_interpretability_model(
    interpretability_model_id,
    interpretability_model,
    interpretability_model_io,
    subject_model,
    subject_model_io,
    task,
    trainer,
    validate_on_non_frozen=True,
    subject_model_count=100,
    batch_size=2**5,
    device="cuda",
):
    """
    Evaluates an interpretability model on the specified subject models.

    validate_on_non_frozen: If set to true, validates only on non-frozen models.
    """
    if not validate_on_non_frozen:
        raise NotImplementedError

    validation_models, _ = get_matching_subject_models_names(
        subject_model_io, trainer, task=task, frozen=NON_FROZEN_ONLY
    )
    validation_models = random.sample(validation_models, subject_model_count)
    print(validation_models)
    validation_dataset = MultifunctionSubjectModelDataset(
        subject_model_io,
        validation_models,
        task,
        subject_model,
        normalise=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    interpretability_model = interpretability_model_io.get_model(
        interpretability_model, interpretability_model_id, device=device
    )
    _, accuracy = _evaluate(
        interpretability_model, validation_dataloader, device=device
    )
    print(f"Evaluation accuracy: {accuracy}")


FROZEN_ONLY = "frozen_only"
NON_FROZEN_ONLY = "non_frozen_only"
BOTH = "both"


def get_matching_subject_models_names(
    model_writer, trainer, task, exclude=[], frozen=BOTH
):
    """
    Returns a list of subject model names that match the specified trainer and task by searching the index.txt file.
    """
    matching_subject_models_names = []
    losses = []
    metadata = model_writer.get_metadata()
    for md in metadata:
        if md["task"]["name"] != type(task).__name__:
            continue

        trainer_metadata = trainer.get_metadata()
        if not all(
            md["trainer"][key] == trainer_metadata[key]
            for key in ["name", "weight_decay", "lr", "l1_penalty_weight"]
        ):
            continue

        if md["id"] in exclude:
            print("exclude")
            continue

        # Have had a few issues where model pickles aren't saved but their
        # metadata is still written, so skip those models.
        if not model_writer.check_model_exists(md["id"]):
            print("does not exist")
            continue

        if frozen == FROZEN_ONLY:
            try:
                if len(md["model"]["frozen"]) == 0:
                    continue
            except KeyError:
                continue
        if frozen == NON_FROZEN_ONLY:
            try:
                if len(md["model"]["frozen"]) != 0:
                    continue
            except KeyError:
                continue

        matching_subject_models_names.append(md["id"])
        losses.append(md["loss"])

    return matching_subject_models_names, sum(losses) / len(losses) if losses else 0


class MultifunctionSubjectModelDataset(Dataset):
    def __init__(
        self,
        model_loader,
        subject_model_ids,
        task,
        subject_model,
        normalise=False,
        device="cpu",
    ):
        self._model_loader = model_loader
        self.subject_model_ids = subject_model_ids
        self.device = device
        self.task = task
        self.subject_model = subject_model

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

        x = torch.concat(
            [param.detach().reshape(-1) for _, param in model.named_parameters()]
        )
        if self._normalise:
            x = (x - self._mean) / self._std

        # x = x[-2740:]

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
        # return torch.cat([x, pe], dim =-1)


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


class FreezableClassifier:
    def __init__(self, file):
        script_dir = os.path.dirname(os.path.abspath(file))
        base_model_path = os.path.join(script_dir, "base_model.pickle")
        checkpoint = torch.load(base_model_path)
        self.load_state_dict(checkpoint)
