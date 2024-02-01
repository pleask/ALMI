from itertools import combinations
import os
import random
from auto_mi.io import DirModelWriter, TarModelWriter

from sklearn import datasets
import torch.nn.functional as F
import torch.nn as nn

from auto_mi.base import MetadataBase
from auto_mi.tasks import SimpleTask, SimpleExample, TRAIN, MI
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
    def __init__(self, *_, variant=0): 
        """
        Variant 70 is the smallest subject modelscontrol show job <job_id>.
        """
        super().__init__()

        conv_channel_variants = list(range(20, 30))
        linear_width_variants = list(range(30, 40))
        variants = [(a, b) for a in conv_channel_variants for b in linear_width_variants]   

        rng = random.Random(42)
        rng.shuffle(variants)
        self.variant = variant

        self._conv_channels, self._linear_width = variants[self.variant]

        self.conv1 = nn.Conv2d(1, self._conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self._conv_channels, self._conv_channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(self._conv_channels * 2 * 2, self._linear_width)
        self.fc2 = nn.Linear(self._linear_width, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, self._conv_channels * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_metadata(self):
        md = super().get_metadata()
        md['variant'] = self.variant
        return md


if __name__ == '__main__':
    train_cli(
        ['sklearn_digits'],
        TarModelWriter,
        DirModelWriter,
        PermutedDigitsTask,
        DigitsClassifier,
        default_subject_model_epochs=100,
        default_subject_model_batch_size=1000,
        default_subject_model_lr=0.01,
        default_interpretability_model_num_layers=8,
        default_interpretability_model_num_heads=8,
        default_interpretability_model_positional_encoding_size=2048,
    )