from itertools import combinations
import os
import random
import torch.nn.functional as F
import torch.nn as nn

from keras.datasets import cifar10

from auto_mi.io import DirModelWriter
from auto_mi.base import MetadataBase
from auto_mi.tasks import SimpleTask, SimpleExample, TRAIN, MI
from auto_mi.subject_models import FreezableClassifier
from auto_mi.cli import train_cli


TRAIN_RATIO = 0.7


class PermutedCIFARTask(SimpleTask):
    def __init__(self, **kwargs):
        super().__init__(
            CIFAR10Example,
            (
                3,  # 3 channels
                32, # 32x32 image
                32, 
            ),
            **kwargs
        )


class CIFAR10Example(SimpleExample):
    def _get_dataset(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        X = x_train 
        y = y_train.reshape(-1) # flatten
        return X, y

    def __getitem__(self, i):
        X, y = super().__getitem__(i)
        return X.reshape(3, 32, 32), y


class CIFAR10Classifier(nn.Module, MetadataBase):
    def __init__(self, *_, variant=0):
        super().__init__()

        conv_channel_variants = list(range(20, 30))
        linear_width_variants = list(range(30, 40))
        variants = [
            (a, b) for a in conv_channel_variants for b in linear_width_variants
        ]

        if variant >= 0:
            rng = random.Random(42)
            rng.shuffle(variants)
        else:
            variant = 0
        self.variant = variant

        self._conv_channels, self._linear_width = variants[self.variant]

        self.conv1 = nn.Conv2d(3, self._conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            self._conv_channels, self._conv_channels, kernel_size=3, padding=1
        )
        self.fc1 = nn.Linear(self._conv_channels * 8 * 8, self._linear_width) 
        self.fc2 = nn.Linear(self._linear_width, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, self._conv_channels * 8 * 8) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_metadata(self):
        md = super().get_metadata()
        md["variant"] = self.variant
        return md


if __name__ == "__main__":
    train_cli(
        ["cifar_10"],
        DirModelWriter,
        DirModelWriter,
        PermutedCIFARTask,
        CIFAR10Classifier,
        # FreezableDigitsClassifier,
        default_subject_model_epochs=100,
        default_subject_model_batch_size=1000,
        default_subject_model_lr=0.01,
        default_interpretability_model_num_layers=8,
        default_interpretability_model_num_heads=8,
        default_interpretability_model_positional_encoding_size=2048,
    )
