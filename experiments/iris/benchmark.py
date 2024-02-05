import argparse
import random
from auto_mi.io import DirModelWriter

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import wandb

from auto_mi.base import MetadataBase
from auto_mi.tasks import TRAIN, VAL, SimpleTask
from auto_mi.tasks import SimpleExample
from auto_mi.cli import train_cli


class PermutedIrisTask(SimpleTask):
    def __init__(self, seed=0., train=True, **kwargs):
        super().__init__(PermutedIrisExample, (4,), num_classes=3, seed=seed, train=train)


class PermutedIrisExample(SimpleExample):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_dataset(self):
        iris_dataset = datasets.load_iris()
        X = iris_dataset.data.astype(float)
        y = iris_dataset.target

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y


class IrisClassifier(nn.Module, MetadataBase):
    def __init__(self, *_, **__):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


if __name__ == '__main__':
    train_cli(
        ['irises'],
        DirModelWriter,
        DirModelWriter,
        PermutedIrisTask,
        IrisClassifier,
        default_subject_model_epochs=300,
        default_interpretability_model_num_layers=1,
        default_interpretability_model_num_heads=16,
        default_interpretability_model_positional_encoding_size=64,
    )