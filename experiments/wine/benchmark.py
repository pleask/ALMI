from itertools import permutations
import os

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset

from auto_mi.base import MetadataBase
from auto_mi.tasks import Task, Example, TRAIN
from auto_mi.utils import DirModelWriter
from auto_mi.cli import train_cli


TRAIN_RATIO = 0.7


class PermutedWineTask(Task):
    def __init__(self, seed=0., train=True, **kwargs):
        super().__init__(seed=seed, train=train)
        self._permutations = list(permutations(range(3)))

    def get_dataset(self, i, type=TRAIN, **_) -> Dataset:
        """
        Gets the dataset for the ith example of this task.
        """
        return PermutedWineExample(self._permutations[i % len(self._permutations)], type=type)

    @property
    def input_shape(self):
        return (13,)

    @property
    def output_shape(self):
        return (3, )

    @property
    def mi_output_shape(self):
        return (3, 3)

    def criterion(self, x, y):
        return F.nll_loss(x, y)


# TODO: Commonise this with the iris code (same for models)
class PermutedWineExample(Example):
    def __init__(self, permutation_map, type=TRAIN):
        self._permutation_map = permutation_map

        wine_dataset = datasets.load_wine()
        X = wine_dataset.data
        y = wine_dataset.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        if type == TRAIN:
            self.X = X_train
            self.y = y_train
        else:
            self.X = X_test
            self.y = y_test

    def __getitem__(self, i):
        x = self.X[i].astype(np.float32)
        y = self.y[i]
        return x, self._permutation_map[y]

    def __len__(self):
        return len(self.X)

    def get_metadata(self):
        return {'permutation_map': self._permutation_map}
    
    def get_target(self):
        return F.one_hot(torch.tensor(self._permutation_map)).to(torch.float32)


class WineClassifier(nn.Module, MetadataBase):
    def __init__(self, *_):
        super().__init__()
        self.fc1 = nn.Linear(13, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


if __name__ == '__main__':
    train_cli(
        DirModelWriter,
        DirModelWriter,
        PermutedWineTask,
        WineClassifier,
        default_subject_model_epochs=100,
        default_subject_model_batch_size=1000,
        default_subject_model_lr=0.01,
        default_interpretability_model_num_layers=2,
        default_interpretability_model_num_heads=16,
        default_interpretability_model_positional_encoding_size=1024,
    )