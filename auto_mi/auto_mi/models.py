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
    def __init__(self, _):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2000, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 2000)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


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