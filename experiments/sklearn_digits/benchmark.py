import argparse
from itertools import permutations
import os
import random

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import wandb

from auto_mi.base import MetadataBase
from auto_mi.rl import pretrain_subject_models
from auto_mi.tasks import Task, Example, TRAIN, VAL
from auto_mi.trainers import AdamTrainer
from auto_mi.utils import DirModelWriter, TarModelWriter
from auto_mi.mi import get_matching_subject_models_names, Transformer, MultifunctionSubjectModelDataset, train_mi_model


TRAIN_RATIO = 0.7


# TODO: commonise with other sklearn tasks
class PermutedDigitsTask(Task):
    def __init__(self, seed=0., train=True, **kwargs):
        super().__init__(seed=seed, train=train)
        self._permutations = list(permutations(range(3)))
        p = list(permutations(range(10)))
        # Shuffle the permutations so we see examples where all output classes
        # are remapped.
        r = random.Random(seed)
        r.shuffle(p)
        self._permutations = p

    def get_dataset(self, i, type=TRAIN, **_) -> Dataset:
        """
        Gets the dataset for the ith example of this task.
        """
        return PermutedDigitsExample(self._permutations[i % len(self._permutations)], type=type)

    @property
    def input_shape(self):
        return (8, 8,)

    @property
    def output_shape(self):
        return (10, )

    @property
    def mi_output_shape(self):
        return (10, 10)

    def criterion(self, x, y):
        return F.nll_loss(x, y)


# TODO: Commonise this with the iris code (same for models)
class PermutedDigitsExample(Example):
    def __init__(self, permutation_map, type=TRAIN):
        self._permutation_map = permutation_map

        digits_dataset = datasets.load_digits()
        X = digits_dataset.data
        y = digits_dataset.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        if type == TRAIN:
            self.X = X_train
            self.y = y_train
        else:
            self.X = X_test
            self.y = y_test

    def __getitem__(self, i):
        x = self.X[i].astype(np.float32)
        y = self.y[i]
        return x.reshape((8, 8)), self._permutation_map[y]

    def __len__(self):
        return len(self.X)

    def get_metadata(self):
        return {'permutation_map': self._permutation_map}
    
    def get_target(self):
        return F.one_hot(torch.tensor(self._permutation_map)).to(torch.float32)


class DigitsClassifier(nn.Module, MetadataBase):
    def __init__(self, *_):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(20 * 2 * 2, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 20 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def evaluate(subject_model_io):
    metadata = subject_model_io.get_metadata()
    accuracies = []
    for model_idx in range(len(metadata)):
        print(f'Model {model_idx}')
        task = PermutedDigitsTask(seed=metadata[model_idx]['task']['seed'])
        example = task.get_dataset(metadata[model_idx]['index'], type=VAL)
        model_id = metadata[model_idx]['id']
        permutation_map = metadata[model_idx]['example']['permutation_map']
        print(f"Permutation map: {permutation_map}")
        model = subject_model_io.get_model(DigitsClassifier(), model_id)
        
        correct = []
        for _ in range(100):
            i = random.randint(0, len(example)-1)
            input, label = example[i]
            prediction = model(torch.Tensor(input).unsqueeze(0))
            # print(label, torch.argmax(prediction, -1).item())
            correct.append((torch.argmax(prediction, -1) == label)[0].item())
        accuracy = sum(correct) /  100
        print(accuracy)
        accuracies.append(accuracy)
    print(sum(accuracies)/ len(accuracies))
    
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run either pretraining or the full pipeline.')
    parser.add_argument("--evaluate_subject_model", action='store_true', help="Evaluate a subject model.")
    parser.add_argument("--train_subject_models", action='store_true', help="Train the subject models.")
    parser.add_argument("--batch_size", type=int, help="Number of subject models to train", default=100)
    parser.add_argument("--overfit", action='store_true', help='Overfit the subject models to the data (ie. make the subject models big)')
    parser.add_argument("--seed", type=float, help="Random seed.", default=0.)
    parser.add_argument("--device", type=str, default="cuda", help="Device to train or evaluate models on")
    parser.add_argument("--subject_model_path", type=str, help="Path of the subject models")
    parser.add_argument("--interpretability_model_path", type=str, help="Path of the interpretability models")
    args = parser.parse_args()

    subject_model_io = DirModelWriter(args.subject_model_path)
    interpretability_model_io = DirModelWriter(args.interpretability_model_path)

    if args.evaluate_subject_model:
        evaluate(subject_model_io)
        quit()

    task = PermutedDigitsTask(args.seed)
    epochs = 100
    trainer = AdamTrainer(task, epochs, 1000, lr=0.01, device=args.device)

    subject_model = DigitsClassifier
    sample_model = subject_model(task)
    subject_model_parameter_count = sum(p.numel() for p in sample_model.parameters())
    print('Layer parameters')
    print(f'Subject model parameter count: {subject_model_parameter_count}', flush=True)

    if args.train_subject_models:
        l1_penalty_weights = [0.]
        state_space = [trainer]

        print('Pretraining subject models')
        trainer = random.choice(state_space)
        pretrain_subject_models(trainer, subject_model_io, subject_model, task, batch_size=args.batch_size)
    else:
        wandb.init(project='bounding-mi', entity='patrickaaleask', reinit=True)

        interpretability_model = Transformer(subject_model_parameter_count, task.mi_output_shape, hidden_size=128, num_layers=4, num_heads=8).to(args.device)
        interpretability_model_parameter_count = sum(p.numel() for p in interpretability_model.parameters())
        print(f'Interpretability model parameter count: {interpretability_model_parameter_count}')
        train_mi_model(interpretability_model, interpretability_model_io, subject_model, subject_model_io, trainer, task, device=args.device)