from abc import ABC, abstractmethod

from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.prune import L1Unstructured

from .base import MetadataBase
from .tasks import to_int


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
            # print(f'Epoch: {epoch}', flush=True)
            for inputs, labels in training_data:
                    optimizer.zero_grad()
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    output = net(inputs)
                    loss = self.task.criterion(output, labels)
                    loss.backward()
                    optimizer.step()
            validation_loss = self.evaluate(net, validation_example)

        validation_loss = self.evaluate(net, validation_example, final=True)

        if self.prune_amount > 0.:
            pruner = L1Unstructured(self.prune_amount)
            for _, param in net.named_parameters():
                param.data = pruner.prune(param.data)

        return validation_loss

    def train_parallel(self, nets, examples, validation_examples):
        optimizers = [optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay) for net in nets]
        training_dataloaders = [DataLoader(example, batch_size=self.batch_size, shuffle=True) for example in examples]

        for epoch in tqdm(range(self.epochs), desc='Subject model epochs'):
            for net, optimizer, training_data in zip(nets, optimizers, training_dataloaders):
                net = net.to(self.device)
                net.train()
                for inputs, labels in training_data:
                    optimizer.zero_grad()
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    output = net(inputs)
                    loss = self.task.criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                net.eval()

        validation_losses = [self.evaluate(net, validation_example, final=True) for net, validation_example in zip(nets, validation_examples)]

        if self.prune_amount > 0.:
            pruner = L1Unstructured(self.prune_amount)
            for net in nets:
                for _, param in net.named_parameters():
                    param.data = pruner.prune(param.data)

        return validation_losses

    
    def evaluate(self, net, example, final=False):
        data = DataLoader(example, batch_size=self.batch_size)
        inputs, labels = next(iter(data))
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs)
        if final:
            for o, l in zip(to_int(outputs[:20]), to_int(labels[:20])):
                # print(round(o.item(), 2), l.item())
                continue
        return self.task.criterion(outputs, labels).detach().cpu().item()

    def get_metadata(self):
        return super().get_metadata() | {
            'weight_decay': self.weight_decay,
            'prune_amount': self.prune_amount,
            'lr': self.lr,
        }
