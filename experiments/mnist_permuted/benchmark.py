from itertools import permutations
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from auto_mi.base import MetadataBase
from auto_mi.tasks import Task, Example, TRAIN
from auto_mi.cli import train_cli
from auto_mi.io import TarModelWriter, DirModelWriter


TRAIN_RATIO = 0.7


class PermutedMNISTTask(Task):
    """
    The permuted MNIST interpretabilty task consists of training models to
    perform MNIST classification but with the output labels permuted (eg. 3 maps
    to 7). The interpretabilty task is figuring out what the permutation is for
    each subject model.
    """
    def __init__(self, seed=0., train=True, **kwargs):
        super().__init__(seed=seed, train=train)
        p = list(permutations(range(10)))
        # Shuffle the permutations so we see examples where all output classes
        # are remapped.
        r = random.Random(seed)
        r.shuffle(p)
        if type == TRAIN:
            self._permutations = p[:int(TRAIN_RATIO * len(p))]
        else:
            self._permutations = p[int(TRAIN_RATIO * len(p)):]

    def get_dataset(self, i, type=TRAIN, **_) -> Dataset:
        """
        Gets the dataset for the ith example of this task.
        """
        return PermutedMNISTExample(self._permutations[i], type=type)

    @property
    def input_shape(self):
        return (28, 28)

    @property
    def output_shape(self):
        return (10, )

    @property
    def mi_output_shape(self):
        return (10, 10)

    def criterion(self, x, y):
        return F.nll_loss(x, y)


class PermutedMNISTExample(Example):
    def __init__(self, permutation_map, type=TRAIN):
        self._permutation_map = permutation_map
        if type == TRAIN:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            self._dataset = torchvision.datasets.MNIST(root='./data', train=type==TRAIN, transform=transform, download=True)

    def __getitem__(self, i):
        img, target = self._dataset[i]
        return img, self._permutation_map[target]

    def __len__(self):
        return len(self._dataset)

    def get_metadata(self):
        return {'permutation_map': self._permutation_map}
    
    def get_target(self):
        return F.one_hot(torch.tensor(self._permutation_map)).to(torch.float32)


class MNIST_CNN(nn.Module, MetadataBase):
    def __init__(self, *_):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(4*7*7, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 8)
        self.fc4 = nn.Linear(8, 8)
        self.fc5 = nn.Linear(8, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 4*7*7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    train_cli(
        ['mnist_permuted'],
        TarModelWriter,
        DirModelWriter,
        PermutedMNISTTask,
        MNIST_CNN,
        default_subject_model_epochs=30,
        default_subject_model_batch_size=1000,
        default_subject_model_lr=0.01,
        default_interpretability_model_num_layers=1,
        default_interpretability_model_num_heads=2,
        default_interpretability_model_positional_encoding_size=2048,
        validate_on_non_frozen=True
    )