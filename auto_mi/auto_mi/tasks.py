from abc import ABC, abstractmethod
from itertools import permutations

import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset

from .base import MetadataBase

TRAIN = "train"
VAL = "val"
MI = "mi"


class Task(MetadataBase, ABC):
    def __init__(self, seed=0.0, train=True, num_classes=-1, num_examples=-1, **_):
        """
        seed: The seed to use for randomly generating examples of this task.
        """
        self.seed = seed
        self.train = train
        self.num_classes = num_classes
        self.num_examples = num_examples

    @abstractmethod
    def get_dataset(self, i, type=TRAIN) -> Dataset:
        """
        Gets the dataset for the ith example of this task.
        """
        pass

    @property
    @abstractmethod
    def input_shape(self):
        pass

    @property
    @abstractmethod
    def output_shape(self):
        pass

    @property
    @abstractmethod
    def mi_output_shape(self):
        pass

    def get_metadata(self):
        return super().get_metadata() | {
            "seed": self.seed,
            "num_classes": self.num_classes,
            "num_examples": self.num_examples,
        }


class Example(MetadataBase, Dataset, ABC):
    @abstractmethod
    def get_target(self):
        pass


class SimpleTask(Task):
    def __init__(
        self,
        example_class,
        input_shape,
        num_examples=-1,
        num_classes=-1,
        seed=0.0,
        train=True,
        **kwargs
    ):
        super().__init__(
            seed=seed, train=train, num_classes=num_classes, num_examples=num_examples
        )
        self._input_shape = input_shape
        self.permutations = list(permutations(range(num_classes)))
        self.example_class = example_class

    def get_dataset(self, i, type=TRAIN, **_):
        return self.example_class(
            self.permutations[i % len(self.permutations)],
            type=type,
            num_examples=self.num_examples,
            num_classes=self.num_classes,
        )

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
    def __init__(self, permutation_map, type=TRAIN, num_examples=-1, num_classes=-1):
        self._permutation_map = permutation_map

        if type == MI:
            return

        X, y = self._get_dataset()

        # Filter out the examples where the class is not one of the first num_classes
        if num_classes != -1:
            X = X[y < num_classes]
            y = y[y < num_classes]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Only use the first num_examples to train
        X_train = X_train[:num_examples]
        y_train = y_train[:num_examples]

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
        return {"permutation_map": self._permutation_map}

    def get_target(self):
        return F.one_hot(torch.tensor(self._permutation_map)).to(torch.float32)
