from abc import ABC, abstractmethod
from functools import partial
import random 

import torch
from torch.utils.data import Dataset
import torch.nn as nn

from .base import MetadataBase

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

class Task(MetadataBase, ABC):
    def __init__(self, seed=0.):
        """
        seed: The seed to use for randomly generating examples of this task.
        """
        self.seed = seed

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
        return super().get_metadata().update({'seed': self.seed})


FUNCTION_NAMES = [
        'addition',
        'multiplication',
        'sigmoid',
        'exponent',
        'min',
]

class SimpleFunctionRecoveryTask(Task):
    criterion = nn.MSELoss()

    def get_dataset(self, i, type=TRAIN):
        random_generator = random.Random(self.seed + i)
        fn_name = random_generator.choice(FUNCTION_NAMES)
        param = random_generator.random()

        seed = param
        if type == VAL:
            seed += 1
        elif type == TEST:
            seed += 2

        return SimpleFunctionRecoveryExample(fn_name, param, seed) 

    @property
    def mi_output_shape(self):
        return (6, )

    @property
    def input_shape(self):
        return (1, )

    @property
    def output_shape(self):
        return (1, )


class Example(Dataset, ABC):
    @abstractmethod
    def get_metadata(self):
        pass

    @abstractmethod
    def get_target(self):
        pass

    
class SimpleFunctionRecoveryExample(Example):
    size = 100000

    def __init__(self, fn_name, param, seed):
        self.fn_name = fn_name
        self.param = param
        self.function = self.get_subject_fn(fn_name, param)
        self.seed = seed

    def __getitem__(self, i):
        random_generator = random.Random(self.seed + i)
        x = torch.tensor([random_generator.random()])
        y = self.function(x)
        return x, y

    def __len__(self):
        return self.size

    def get_metadata(self):
        return {'fn_name': self.fn_name, 'param': self.param}

    def get_subject_fn(self, fn_name, param):
        """
        Returns a torch function that implements the specified function.

        The functions map onto the range [0, 100] (give or take).

        fn_name: the name of the function
        param: a float between 0 and 1
        """
        if fn_name == FUNCTION_NAMES[0]:
            return partial(lambda c, x: (x + c)/2, param)
        elif fn_name == FUNCTION_NAMES[1]:
            return partial(lambda c, x: x * c * 10, param)
        elif fn_name == FUNCTION_NAMES[2]:
            return partial(lambda c, x: 20*(1/(1+torch.exp(-(x+c)))-0.5), param)
        elif fn_name == FUNCTION_NAMES[3]:
            return partial(lambda c, x: x ** (c / 2), param)
        elif fn_name == FUNCTION_NAMES[4]:
            return partial(lambda c, x: torch.min(torch.full_like(x, c), x), param)
        else:
            raise ValueError(f'Invalid function name: {fn_name}')
    
    def get_target(self):
        one_hot = [0.] * len(FUNCTION_NAMES)
        one_hot[FUNCTION_NAMES.index(self.fn_name)] = 1.
        one_hot.append(self.param)
        return torch.tensor(one_hot)