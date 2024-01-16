from itertools import permutations, combinations
import os
import random
from auto_mi.io import DirModelWriter

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset

from auto_mi.base import MetadataBase
from auto_mi.tasks import SimpleTask, SimpleExample, TRAIN, MI
from auto_mi.mi import FreezableClassifier
from auto_mi.cli import train_cli


TRAIN_RATIO = 0.7

class PermutedDigitsTask(SimpleTask):
    def __init__(self, **kwargs):
        super().__init__(PermutedDigitsExample, (8, 8,), **kwargs)


class PermutedDigitsExample(SimpleExample):
    def _get_dataset(self):
        digits_dataset = datasets.load_digits()
        X = digits_dataset.data
        y = digits_dataset.target
        return X, y

    def __getitem__(self, i):
        X, y = super().__getitem__(i)
        return X.reshape(8, 8), y


class DigitsClassifier(nn.Module, MetadataBase):
    def __init__(self, *_):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(20 * 2 * 2, 30)
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 20 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class FreezableDigitsClassifier(DigitsClassifier, FreezableClassifier):
    def __init__(self, *_):
        """
        Any two layers, or a single layer, can be frozen without significantly
        effecting the performance. 
        """
        DigitsClassifier.__init__(self)
        FreezableClassifier.__init__(self, __file__)
        frozen_combinations = list(combinations(list(range(4)), 2)) + list(combinations(list(range(4)), 1)) + [tuple()]
        self.frozen = frozen_combinations[int(os.environ.get('FROZEN', 0))]

    def get_metadata(self):
        md = super().get_metadata()
        md['frozen'] = self.frozen
        return md


if __name__ == '__main__':
    train_cli(
        ['sklearn_digits'],
        DirModelWriter,
        DirModelWriter,
        PermutedDigitsTask,
        # DigitsClassifier,
        FreezableDigitsClassifier,
        default_subject_model_epochs=100,
        default_subject_model_batch_size=1000,
        default_subject_model_lr=0.01,
        default_interpretability_model_num_layers=1,
        default_interpretability_model_num_heads=2,
        default_interpretability_model_positional_encoding_size=2048,
    )