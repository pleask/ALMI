from abc import ABC, abstractmethod
from itertools import permutations

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

from auto_mi.tasks import Task, Example, TRAIN, VAL

class SimpleTask(Task):
    def __init__(self, example_class, input_shape, num_classes, seed=0., train=True, **kwargs):
        super().__init__(seed=seed, train=train)
        self._input_shape = input_shape
        self.num_classes = num_classes
        self.permutations = list(permutations(range(num_classes)))
        self.example_class = example_class

    def get_dataset(self, i, type=TRAIN, **_):
        return self.example_class(self.permutations[i % len(self.permutations)], type=type)

    @property
    def output_shape(self):
        return (self.num_classes,)

    @property
    def mi_output_shape(self):
        return (self.num_classes, self.num_classes)

    @property
    def input_shape(self):
        return self._input_shape

    def criterion(self, x, y):
        return F.nll_loss(x, y)


class SimpleExample(Example, ABC):
    def __init__(self, permutation_map, type=TRAIN):
        self._permutation_map = permutation_map

        X, y = self._get_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if type == TRAIN:
            self.X = X_train
            self.y = y_train
        else:
            self.X = X_test
            self.y = y_test

    @abstractmethod
    def _get_dataset(self):
        pass

    def __getitem__(self, i):
        X = self.X[i].astype(np.float32)
        y = self.y[i]
        return X, self._permutation_map[y]

    def __len__(self):
        return len(self.X)

    def get_metadata(self):
        return {'permutation_map': self._permutation_map}
    
    def get_target(self):
        return F.one_hot(torch.tensor(self._permutation_map)).to(torch.float32)