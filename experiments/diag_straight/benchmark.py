import argparse
from random import Random
import random

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import wandb

from auto_mi.sklearn import SimpleExample, SimpleTask
from auto_mi.base import MetadataBase
from auto_mi.rl import pretrain_subject_models
from auto_mi.tasks import Task, Example, TRAIN, VAL, MI
from auto_mi.trainers import AdamTrainer
from auto_mi.utils import DirModelWriter, TarModelWriter, evaluate_subject_model
from auto_mi.mi import Transformer, train_mi_model, PositionalEncoding, LSTMClassifier


diagonal = [
    np.array([[1., 0.], [0., 1.]]),
    np.array([[0., 1.], [1., 0.]]),
]
straight = [
    np.array([[1., 1.], [0., 0.]]),
    np.array([[0., 0.], [1., 1.]]),
    np.array([[1., 0.], [1., 0.]]),
    np.array([[0., 1.], [0., 1.]]),
]
DIAG = 'diagonal'
STR = 'straight'
OPTIONS = [DIAG, STR]

def get_item(i):
    r = Random(i)
    arr = np.zeros((4, 4))
    # Make sure we get numbers from 0 to 8

    count = 0
    for i in range(2):
        straight_or_diag = r.choice(OPTIONS)
        symbol = None
        if straight_or_diag == DIAG:
            symbol = r.choice(diagonal)
            count += 1
        else:
            symbol = r.choice(straight)
            count -= 1
        arr[i%2*2:i%2*2 + 2] = np.kron(symbol, np.ones((1, 2)))
    if count == 0:
        label = 0
    else:
        label = 1
    return arr, label


class DiagStraightTask(SimpleTask):
    def __init__(self, seed=0., train=True, **kwargs):
        super().__init__(DiagStraightExample, (4, 4), 2, seed=seed, train=train)


class DiagStraightExample(SimpleExample):
    def __init__(self, permutation_map, type=TRAIN, **kwargs):
        super().__init__(permutation_map, type)

    def _get_dataset(self):
        X, Y = [], []
        for i in range(10000):
            x, y = get_item(i)
            X.append(x)
            Y.append(y)

        return X, Y 


class DiagStraightClassifier(nn.Module, MetadataBase):
    def __init__(self,  *_, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=2)
        self.fc1 = nn.Linear(90, 20)
        self.fc2 = nn.Linear(20, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x), 2)
        x = x.view(len(x), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class LinearDiagStraightClassifier(nn.Module, MetadataBase):
    def __init__(self, *_):
        super().__init__()
        self.fc1 = nn.Linear(16, 30)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(30, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.flatten(len(x), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='Run either pretraining or the full pipeline.')
    parser.add_argument("--evaluate_subject_model", action='store_true', help="Evaluate a subject model.")
    parser.add_argument("--train_subject_models", action='store_true', help="Train the subject models.")
    parser.add_argument("--batch_size", type=int, help="Number of subject models to train", default=100)
    parser.add_argument("--linear", action='store_true', help="Use the linear model instead of the convolutional model.")
    parser.add_argument("--overfit", action='store_true', help='Overfit the subject models to the data (ie. make the subject models big)')
    parser.add_argument("--seed", type=float, help="Random seed.", default=0.)
    parser.add_argument("--device", type=str, default="cuda", help="Device to train or evaluate models on")
    parser.add_argument("--subject_model_path", type=str, help="Path of the subject models")
    parser.add_argument("--interpretability_model_path", type=str, help="Path of the interpretability models")
    parser.add_argument("--interpretability_batch_size", type=int, help="Batch size for interpretability model", default=2**8)
    parser.add_argument("--interpretability_mixed_precision", action='store_true', help="Use mixed precision (float16) when training interpretability model.")
    parser.add_argument("--interpretability_gradient_accumulation", type=int, default=1, help="Frequently with which to accumulate gradients when training interpretability model.")
    parser.add_argument("--interpretability_subject_model_count", type=int, default=-1, help="How many subject models to use.")
    args = parser.parse_args()

    subject_model_io = DirModelWriter(args.subject_model_path)
    interpretability_model_io = DirModelWriter(args.interpretability_model_path)

    subject_model_class = DiagStraightClassifier
    if args.evaluate_subject_model:
        evaluate_subject_model(DiagStraightTask, subject_model_class, subject_model_io)
        quit()

    task = DiagStraightTask(args.seed)
    epochs = 100
    trainer = AdamTrainer(task, epochs, 1000, lr=0.01, device=args.device)

    sample_model = subject_model_class(task)
    subject_model_parameter_count = sum(p.numel() for p in sample_model.parameters())
    print('Layer parameters')
    print(f'Subject model parameter count: {subject_model_parameter_count}', flush=True)

    if args.train_subject_models:
        l1_penalty_weights = [0.]
        state_space = [trainer]

        print('Pretraining subject models')
        trainer = random.choice(state_space)
        pretrain_subject_models(trainer, subject_model_io, subject_model_class, task, batch_size=args.batch_size)
    else:
        wandb.init(project='bounding-mi', entity='patrickaaleask', reinit=True, tags=['diag_straight'])

        interpretability_model = Transformer(subject_model_parameter_count, task.mi_output_shape, num_layers=2, num_heads=16, positional_encoding_size=1024).to(args.device)
        interpretability_model_parameter_count = sum(p.numel() for p in interpretability_model.parameters())
        print(f'Interpretability model parameter count: {interpretability_model_parameter_count}')
        train_mi_model(interpretability_model, interpretability_model_io, subject_model_class, subject_model_io, trainer, task, device=args.device, batch_size=args.interpretability_batch_size, amp=args.interpretability_mixed_precision, grad_accum_steps=args.interpretability_gradient_accumulation, subject_model_count=args.interpretability_subject_model_count)