import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MetadataBase


class SimpleFunctionRecoveryModel(nn.Module, MetadataBase):
    def __init__(self, task):
        super().__init__()

        hidden_layer_size = 250
        self.l1 = nn.Linear(task.input_shape[0], hidden_layer_size)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.r2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_layer_size, task.output_shape[0])

    def forward(self, x):
        return self.l3(self.r2(self.l2(self.r1(self.l1(x)))))


class ConvMNIST(nn.Module, MetadataBase):
    def __init__(self, task):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 1)  # 1 input channel, 32 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(4, 4, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(576, 16)
        self.fc2 = nn.Linear(16, 11)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.LogSoftmax(dim=1)(x)
        return output


class IntegerGroupFunctionRecoveryModel(nn.Module, MetadataBase):
    def __init__(self, task):
        super().__init__()

        flattened_input_size = math.prod(task.input_shape)
        hidden_layer_size = 1000
        self.l1 = nn.Linear(flattened_input_size, hidden_layer_size)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_layer_size, task.output_shape[0])

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.l1(x)
        x = self.r1(x)
        x = self.l2(x)
        return torch.sigmoid(x)



SUBJECT_MODELS = {
    "SimpleFunctionRecoveryModel": SimpleFunctionRecoveryModel
}