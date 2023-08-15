from abc import ABC, abstractmethod

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.prune import L1Unstructured

from .base import MetadataBase


class BaseTrainer(MetadataBase, ABC):
    @abstractmethod
    def train(self, model, example):
        """
        Trains a new model on the given example.
        """
        pass


class AdamTrainer(BaseTrainer):
    def __init__(self, task, epochs, batch_size, lr=0.01, weight_decay=0., prune_amount=0., device=torch.device('cpu')):
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.weight_decay = weight_decay
        self.prune_amount = prune_amount
        self.lr = lr
    
    def train(self, net, example, validation_example):
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        training_data = DataLoader(example, batch_size=self.batch_size)

        for epoch in range(self.epochs):
            print(f'Epoch: {epoch}', flush=True)
            for inputs, labels in training_data:
                    optimizer.zero_grad()
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    output = net(inputs)
                    loss = self.task.criterion(output, labels)
                    loss.backward()
                    optimizer.step()
            print('Training loss:', loss.item(), flush=True)
            validation_loss = self.evaluate(net, validation_example)
            print('Validation loss:', validation_loss)

        validation_loss = self.evaluate(net, validation_example, final=True)
        print('Final Validation loss:', validation_loss)

        if self.prune_amount > 0.:
            pruner = L1Unstructured(self.prune_amount)
            for _, param in net.named_parameters():
                param.data = pruner.prune(param.data)

        return validation_loss
    
    def evaluate(self, net, example, final=False):
        data = DataLoader(example, batch_size=self.batch_size)
        inputs, labels = next(iter(data))
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs)
        if final:
            for x, y in zip(outputs[:20].detach().cpu().numpy(), labels[:20].detach().cpu().numpy()):
                print('value', int(''.join(map(str, np.round(x).astype(int))), 2), 'label', int(''.join(map(str, np.round(y).astype(int))), 2))
        return self.task.criterion(outputs, labels).detach().cpu().item()

    def get_metadata(self):
        return super().get_metadata() | {
            'weight_decay': self.weight_decay,
            'prune_amount': self.prune_amount,
            'lr': self.lr,
        }
