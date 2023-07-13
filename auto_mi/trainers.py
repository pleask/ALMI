from abc import ABC, abstractmethod

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.prune import L1Unstructured

class BaseTrainer(ABC):
    @abstractmethod
    def train(self, model, example):
        """
        Trains a new model on the given example.
        """
        pass


class AdamTrainer(BaseTrainer):
    def __init__(self, task, epochs, batch_size, device=torch.device('cpu')):
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def _get_optimiser(self, net):
        return optim.Adam(net.parameters(), lr=0.01)
    
    def train(self, net, example):
        optimizer = self._get_optimiser(net)
        training_data = DataLoader(example, batch_size=self.batch_size)

        for epoch in range(self.epochs):
            print(f'Epoch: {epoch}')
            for inputs, labels in training_data:
                    optimizer.zero_grad()
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    output = net(inputs)
                    loss = self.task.criterion(output, labels)
                    loss.backward()
                    optimizer.step()
    
    def evaluate(self, net, example):
        data = DataLoader(example, batch_size=self.batch_size)
        inputs, labels = next(iter(data))
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs)
        return self.task.criterion(outputs, labels).detach().cpu().item()


class AdamWeightDecayTrainer(AdamTrainer):
    def __init__(self, task, epochs, batch_size, device=torch.device('cpu'), weight_decay=0.0001):
        super().__init__(task, epochs, batch_size, device=device)
        self.weight_decay = weight_decay
    
    def _get_optimiser(self, net):
        return optim.Adam(net.parameters(), lr=0.01, weight_decay=self.weight_decay)


class AdamL1UnstructuredPruneTrainer(AdamTrainer):
    def __init__(self, task, epochs, batch_size, device=torch.device('cpu'), prune_amount=0.1):
        super().__init__(task, epochs, batch_size, device=device)
        self.pruner = L1Unstructured(prune_amount)

    def train(self, net, example):
        super().train(net, example)
        pruner = self.pruner
        for _, param in net.named_parameters():
            param.data = pruner.prune(param.data)
