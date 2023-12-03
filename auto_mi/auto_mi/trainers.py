from abc import ABC, abstractmethod

from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.prune import L1Unstructured

from .base import MetadataBase


class BaseTrainer(MetadataBase, ABC):
    """
    Implements the training loop based on passed optimisers.
    """
    def __init__(self, task, epochs, batch_size):
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size

    @abstractmethod
    def train_parallel(self, models, examples, validation_examples):
        """
        Trains a new model on the given example.
        """
        pass

    def _train_inner(self, models, examples, validation_examples, optimizers, training_dataloaders, l1_penalty_weight=0.):
        train_losses = []
        for i in tqdm(range(self.epochs), desc='Subject model epochs'):
            for net, optimizer, training_data in zip(models, optimizers, training_dataloaders):
                net = net.to(self.device)
                net.train()
                for inputs, labels in training_data:
                    optimizer.zero_grad()
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    output = net(inputs)
                    loss = self.task.criterion(output, labels)
                    if l1_penalty_weight:
                        loss += l1_penalty_weight * sum([torch.abs(param).sum() for param in net.parameters()])
                    loss.backward()
                    optimizer.step()
                net.eval()
                if i == self.epochs - 1:
                    train_losses.append(loss.detach().cpu().item())

        return [self.evaluate(net, validation_example) for net, validation_example in zip(models, validation_examples)], train_losses

    def evaluate(self, net, example):
        data = DataLoader(example, batch_size=self.batch_size)
        inputs, labels = next(iter(data))
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs)
        return self.task.criterion(outputs, labels).detach().cpu().item()


class AdamTrainer(BaseTrainer):
    def __init__(self, task, epochs, batch_size, lr=0.01, weight_decay=0., l1_penalty_weight=0., device=torch.device('cpu')):
        super().__init__(task, epochs, batch_size)
        self.device = device
        self.weight_decay = weight_decay
        self.lr = lr
        self.l1_penalty_weight = l1_penalty_weight

    def train_parallel(self, nets, examples, validation_examples):
        optimizers = [optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay) for net in nets]
        training_dataloaders = [DataLoader(example, batch_size=self.batch_size, shuffle=True) for example in examples]

        validation_losses = self._train_inner(nets, examples, validation_examples, optimizers, training_dataloaders)

        return validation_losses

    def get_metadata(self):
        return super().get_metadata() | {
            'weight_decay': self.weight_decay,
            'lr': self.lr,
            'l1_penalty_weight': self.l1_penalty_weight,
        }
