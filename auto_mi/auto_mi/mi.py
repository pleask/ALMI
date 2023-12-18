import random

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm

from auto_mi.tasks import MI
from auto_mi.base import MetadataBase
from auto_mi.tasks import SimpleFunctionRecoveryTask

VAL_RATIO = 0.2
TRAIN_RATIO = 0.8
INTERPRETABILITY_BATCH_SIZE = 2**7

# TODO: subject_model can go into the IO class rather than be passed in here
def train_mi_model(interpretability_model, interpretability_model_io, subject_model, subject_model_io, trainer, task, batch_size=2**7, epochs=1000, device='cuda', lr=1e-6, amp=False, grad_accum_steps=1, subject_model_count=-1):
    """
    amp: Use automatic mixed precision
    """
    all_subject_models, _ = get_matching_subject_models_names(subject_model_io, trainer, task=task)
    if subject_model_count > 0:
        all_subject_models = all_subject_models[:subject_model_count]
    if len(all_subject_models) == 0:
        raise ValueError('No subject models found')
    wandb.log({'subject_model_count': len(all_subject_models)})
    print(f'Using {len(all_subject_models)} subject models')
    validation_models, train_models  = all_subject_models[:int(VAL_RATIO*len(all_subject_models))], all_subject_models[int(VAL_RATIO*len(all_subject_models)):]

    train_dataset = MultifunctionSubjectModelDataset(subject_model_io, train_models, task, subject_model, normalise=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_dataset = MultifunctionSubjectModelDataset(subject_model_io, validation_models, task, subject_model, normalise=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    optimizer = torch.optim.Adam(interpretability_model.parameters(), lr=lr)#, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1000, factor=0.1, verbose=True)
    criterion = nn.CrossEntropyLoss()

    scaler = None
    if amp:
        scaler = GradScaler()

    if grad_accum_steps > 1:
        total_loss = 0.

    for epoch in tqdm(range(epochs), desc='Interpretability model epochs'):
        interpretability_model.train()
        for i, (inputs, targets) in enumerate(train_dataloader):
            # TODO: make it so amp and grad accum can be used together
            if amp:
                optimizer.zero_grad()

                with autocast():
                    outputs = interpretability_model(inputs.to(device, non_blocking=True))
                    loss = criterion(outputs, targets.to(device, non_blocking=True))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scheduler.step(loss)
                scaler.update()
                wandb.log({'train_loss': loss})
            elif grad_accum_steps > 1:
                outputs = interpretability_model(inputs.to(device, non_blocking=True))
                loss = 0
                for i in range(outputs.shape[1]):
                    loss += criterion(outputs[:, i], targets[:, i].to(device))
                # loss = criterion(outputs, targets.to(device, non_blocking=True))
                loss.backward()
                total_loss += loss
                wandb.log({'train_loss': loss.detach().cpu()})

                if (i + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    scheduler.step(total_loss)
                    optimizer.zero_grad()
                    total_loss = 0.
            else:
                optimizer.zero_grad()
                outputs = interpretability_model(inputs.to(device, non_blocking=True))
                loss = 0
                for i in range(outputs.shape[1]):
                    loss += criterion(outputs[:, i], targets[:, i].to(device))
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                wandb.log({'train_loss': loss})

        interpretability_model.eval()
        with torch.no_grad():
            eval_loss = 0.
            for inputs, targets in validation_dataloader:
                outputs = interpretability_model(inputs.to(device))
                loss = criterion(outputs, targets.to(device))
                eval_loss += loss.item()
            
            predicted_classes = torch.argmax(outputs, dim=-1)
            target_classes = torch.argmax(targets, dim=-1)
            accuracy = torch.sum(torch.all(predicted_classes.detach().cpu() == target_classes.cpu(), dim=1)).item() / len(predicted_classes)
            
        wandb.log({'validation_loss': eval_loss, 'validation_accuracy': accuracy})

        interpretability_model_io.write_model(f'{trainer.get_metadata()}', interpretability_model)


# TODO: Add wandb logging of the trained models
def train_interpretability_model(model, task, subject_model_path, validation_subject_models, trainer, interpretabilty_model_io, reuse_count=1000):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.001)
    criterion = nn.MSELoss()
    epochs = 5

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

    interpretabilty_model_io.write_model(f'{trainer.get_metadata()}', model)

    model.eval()
    with torch.no_grad():
        eval_loss = 0.
        for inputs, targets in eval_dataloader:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            eval_loss += loss.item()
    
    return eval_loss / len(eval_dataloader), train_loss / len(train_dataloader)


def get_matching_subject_models_names(model_writer, trainer, task=SimpleFunctionRecoveryTask, exclude=[]):
    matching_subject_models_names = []
    losses = []
    metadata = model_writer.get_metadata()
    for md in metadata:
        if md['task']['name'] != type(task).__name__:
            continue

        trainer_metadata = trainer.get_metadata()
        if not all(md['trainer'][key] == trainer_metadata[key] for key in ['name', 'weight_decay', 'lr', 'l1_penalty_weight']):
            continue

        if md['id'] in exclude:
            print('exclude')
            continue

        # Have had a few issues where model pickles aren't saved but their
        # metadata is still written, so skip those models.
        if not model_writer.check_model_exists(md['id']):
            print('does not exist')
            continue

        matching_subject_models_names.append(md['id'])
        losses.append(md['loss'])
    
    return matching_subject_models_names, sum(losses) / len(losses) if losses else 0



class MultifunctionSubjectModelDataset(Dataset):
    def __init__(self, model_loader, subject_model_ids, task, subject_model, normalise=False, device='cpu'):
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

        example = self.task.get_dataset(metadata['index'], type=MI)
        # TODO: Something is broken with the datasets so here's a workaround but fix asap
        example._permutation_map = metadata['example']['permutation_map']
        y = example.get_target()

        model = self._model_loader.get_model(self.subject_model(self.task), name)

        x = torch.concat(
            [param.detach().reshape(-1) for _, param in model.named_parameters()]
        )
        if self._normalise:
            x = (x - self._mean) / self._std

        self._data[idx] = (x, y)

        return x, y

    def _index_metadata(self):
        d = {}
        for md in self._model_loader.get_metadata():
            d[md['id']] = md
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
        div_term = torch.exp(torch.arange(0, encoding_length, 2).float() * -(torch.log(torch.tensor(10000.0)) / encoding_length))

        self.position_weights = nn.Parameter(torch.sin(position * div_term), requires_grad=True)
        self.position_biases = nn.Parameter(torch.cos(position * div_term), requires_grad=True)

    def forward(self, x):
        pe = torch.cat([self.position_weights[:x.size(1), :], self.position_biases[:x.size(1), :]], dim=1).unsqueeze(0)
        pe = pe.expand(x.shape[0], -1, -1)
        return torch.cat([x, pe], dim =-1)


class Transformer(nn.Module, MetadataBase):
    def __init__(self, subject_model_parameter_count, out_shape, num_layers=6, num_heads=1):
        super().__init__()
        self.out_shape = out_shape
        output_size = torch.zeros(out_shape).view(-1).shape[0]
        self.positional_encoding = PositionalEncoding(2048, subject_model_parameter_count)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.positional_encoding.length * 2, nhead=num_heads, dim_feedforward=2048, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.positional_encoding.length * 2),
        )

        # Linear layer for classification
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.positional_encoding.length * 2, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.expand(x.shape[0], x.shape[1], self.positional_encoding.length)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)

        x = x.transpose(1, 2)
        x = self.global_avg_pooling(x).squeeze(2)
        output = self.fc(x)
        output = output.view(-1, *self.out_shape)
        return self.softmax(output)


class LSTMClassifier(nn.Module):
    def __init__(self, subject_model_parameter_count, out_shape, hidden_size=2048):
        super(LSTMClassifier, self).__init__()
        self.out_shape = out_shape
        output_size = torch.zeros(out_shape).view(-1).shape[0]
        self.lstm = nn.LSTM(subject_model_parameter_count, hidden_size, batch_first=True, num_layers=3)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        output = output.view(-1, *self.out_shape)
        return output


class FeedForwardNN(nn.Module, MetadataBase):
    def __init__(self, in_size, out_size, layer_scale=1):
        super().__init__()
        self.fc1 = nn.Linear(in_size, int(self.positional_encoding.length*layer_scale))
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

    @property
    def device(self):
        return self.fc1.weight.device


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
