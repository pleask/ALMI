"""
Methods for training and evaluating interpretability models.
"""
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
    split_on_variants=False,
    variant_range_start=-1,
    variant_range_end=-1,
    num_layers=6,
    num_heads=8,
    positional_encoding_size=2048,
    load_interpretability_model=None,
):
    """
    Trains an interpretability transformer model on the specified subject models.


    If split_on_variants is set to true, the interpretability model is trained
    on the first 70 vairants of the subject models and validated on the last 30
    variants. Otherwise, the interpretability model is trained on the first 70%
    of the subject models and validated on the last 30%, regardless of variant.
    It is assumed that there are 100 variants, and that there are equal numbers
    of each variant.
    """
    if not split_on_variants:
        all_subject_models, _ = get_matching_subject_models_names(
            subject_model_io,
            trainer,
            task=task,
            variants=list(range(variant_range_start, variant_range_end)),
        )
        if subject_model_count > 0:
            all_subject_models = all_subject_models[:subject_model_count]
        validation_models, train_models = (
            all_subject_models[: int(VAL_RATIO * len(all_subject_models))],
            all_subject_models[int(VAL_RATIO * len(all_subject_models)) :],
        )
    else:
        train_models, _ = get_matching_subject_models_names(
            subject_model_io,
            trainer,
            task=task,
            variant_range=list(range(0, 70)),
        )
        validation_models, _ = get_matching_subject_models_names(
            subject_model_io,
            trainer,
            task=task,
            variants=list(range(70, 100)),
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
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    validation_dataset = MultifunctionSubjectModelDataset(
        subject_model_io,
        train_models,
        task,
        subject_model,
    )
    # Make sure the validation dataset uses the same normalisation as the train
    if train_dataset._normalise:
        validation_dataset._std = train_dataset._std
        validation_dataset._mean = train_dataset._mean
        validation_dataset._normalise = True

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    interpretability_model = Transformer(
        max(train_dataset.max_elements, validation_dataset.max_elements),
        task.mi_output_shape,
        num_layers=num_layers,
        num_heads=num_heads,
        positional_encoding_size=positional_encoding_size,
    ).to(device)
    interpretability_model_parameter_count = sum(
        p.numel() for p in interpretability_model.parameters()
    )
    print(
        f"Interpretability model parameter count: {'{:,}'.format(interpretability_model_parameter_count)}"
    )
    if load_interpretability_model:
        interpretability_model = interpretability_model_io.get_model(
            interpretability_model, load_interpretability_model
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

    for epoch in tqdm(range(50), desc="Interpretability model epochs"):
        interpretability_model.train()
        total_loss = 0.0
        for i, (inputs, masks, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = interpretability_model(
                inputs.to(device, non_blocking=True),
                masks.to(device, non_blocking=True),
            )
            predicted_classes = torch.argmax(outputs, dim=-1)
            loss = 0
            for i in range(outputs.shape[1]):
                loss += CRITERION(outputs[:, i], targets[:, i].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(interpretability_model.parameters(), 0.5)
            optimizer.step()
            wandb.log({"train_loss": loss})
            total_loss += loss
        avg_train_loss = total_loss / len(train_dataloader)

        scheduler.step(loss)
        eval_loss, accuracy = _evaluate(
            interpretability_model, validation_dataloader, device=device
        )
        wandb.log(
            {
                "validation_loss": eval_loss,
                "validation_accuracy": accuracy,
            }
        )
        tqdm.write(
            f"Epoch: {epoch}, Train Loss: {avg_train_loss}, Validation Loss: {eval_loss}, Validation Accuracy: {accuracy}"
        )

        interpretability_model_io.write_model(run.id, interpretability_model)


def _evaluate(interpretability_model, validation_dataloader, device="cuda"):
    eval_loss = 0.0
    accuracy = 0.0

    # TODO: Figure out why setting the model to eval breaks validation
    # interpretability_model.eval()
    with torch.no_grad():
        for inputs, masks, targets in validation_dataloader:
            outputs = interpretability_model(inputs.to(device), masks.to(device))
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
        accuracy /= len(validation_dataloader)

    return eval_loss, accuracy


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

        # Flatten the samples and calculate the maximum number of elements
        samples = [torch.flatten(self[i][0]) for i in range(len(subject_model_ids))]
        self.max_elements = max(len(sample) for sample in samples)

        if normalise:
            params = torch.cat(samples)
            self._std, self._mean = torch.std_mean(params, dim=-1)
            self._normalise = True

        self._data = [None for _ in self.subject_model_ids]

    def __len__(self):
        return len(self.subject_model_ids)

    def __getitem__(self, idx):
        try:
            if self._data[idx]:
                return self._data[idx]
        except AttributeError:
            # The time we run this we don't know what the max length in the data is.
            pass
        name = self.subject_model_ids[idx]
        metadata = self.metadata[name]

        example = self.task.get_dataset(metadata["index"], type=MI)
        y = example.get_target()

        # TODO: Can move the variant code to the loader later on
        try:
            variant = metadata["model"]["variant"]
        except KeyError:
            variant = -1
        try:
            model = self._model_loader.get_model(
                self.subject_model(self.task, variant=variant), name
            )
        except Exception:
            print(name)
            quit()

        x = torch.concat(
            [param.detach().reshape(-1) for _, param in model.named_parameters()]
        )
        mask = torch.ones(len(x))

        try:
            if self._normalise:
                x = (x - self._mean) / self._std

            if len(x) < self.max_elements:
                x = torch.nn.functional.pad(x, (0, self.max_elements - len(x)))
                mask = torch.nn.functional.pad(mask, (0, self.max_elements - len(mask)))

            self._data[idx] = (x, mask, y)
        except AttributeError as e:
            # If we haven't parameterised the normalisation yet, just skip.
            pass

        return x, mask, y

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
    """
    Chunks the input and embeds each chunk separately before passing it through
    the transformer.
    """

    def __init__(
        self,
        subject_model_parameter_count,
        out_shape,
        num_layers=6,
        num_heads=8,
        positional_encoding_size=4096,
        chunk_size=128,
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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.TransformerEncoderLayer):
            nn.init.xavier_uniform_(module.self_attn.in_proj_weight)
            nn.init.kaiming_normal_(module.linear1.weight, nonlinearity="relu")
            nn.init.kaiming_normal_(module.linear2.weight, nonlinearity="relu")

    def _pad_to_factor(self, x):
        padding = self._chunk_size - (x.size(1) % self._chunk_size)
        if padding != self._chunk_size:
            x = torch.nn.functional.pad(x, (0, padding))
        return x

    def _chunk(self, x):
        return x.view(x.size(0), x.size(1) // self._chunk_size, self._chunk_size)

    def _chunk_input(self, x, masks):
        x, masks = self._pad_to_factor(x), self._pad_to_factor(masks)
        x, masks = self._chunk(x), self._chunk(masks)
        masks = (masks.sum(dim=2) != 0).float()
        return x, masks

    def forward(self, x, masks):
        """
        masks represents which values of the inputs were padded.
        """
        x, mask = self._chunk_input(x, masks)
        x = self.embedding(x) * math.sqrt(self.positional_encoding.length)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = x.transpose(1, 2)
        x = self.global_avg_pooling(x).squeeze(2)
        output = self.fc(x)
        output = output.view(-1, *self.out_shape)
        return output
