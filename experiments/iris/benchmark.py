import argparse
import random

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import wandb

from auto_mi.base import MetadataBase
from auto_mi.rl import pretrain_subject_models
from auto_mi.tasks import TRAIN, VAL
from auto_mi.trainers import AdamTrainer
from auto_mi.utils import DirModelWriter, TarModelWriter
from auto_mi.mi import PositionalEncoding, Transformer, train_mi_model
from auto_mi.sklearn import SimpleExample, SimpleTask


class PermutedIrisTask(SimpleTask):
    def __init__(self, seed=0., train=True, **kwargs):
        super().__init__(PermutedIrisExample, (4,), 3, seed=seed, train=train)


class PermutedIrisExample(SimpleExample):
    def __init__(self, permutation_map, type=TRAIN):
        super().__init__(permutation_map, type)

    def _get_dataset(self):
        iris_dataset = datasets.load_iris()
        X = iris_dataset.data.astype(float)
        y = iris_dataset.target

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y


class IrisClassifier(nn.Module, MetadataBase):
    def __init__(self, *_):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


def evaluate(subject_model_io):
    metadata = subject_model_io.get_metadata()
    accuracies = []
    for model_idx in range(len(metadata)):
        print(f'Model {model_idx}')
        task = PermutedIrisTask(seed=metadata[model_idx]['task']['seed'])
        example = task.get_dataset(metadata[model_idx]['index'], type=VAL)
        model_id = metadata[model_idx]['id']
        permutation_map = metadata[model_idx]['example']['permutation_map']
        print(f"Permutation map: {permutation_map}")
        model = subject_model_io.get_model(IrisClassifier(), model_id)
        
        correct = []
        for _ in range(10):
            i = random.randint(0, len(example)-1)
            input, label = example[i]
            prediction = model(torch.Tensor(input).unsqueeze(0))
            # print(label, torch.argmax(prediction, -1).item())
            correct.append((torch.argmax(prediction, -1) == label)[0].item())
        print(correct)
        accuracy = sum(correct) /  10
        accuracies.append(accuracy)
    print(sum(accuracies)/ len(accuracies))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run either pretraining or the full pipeline.')
    parser.add_argument("--evaluate_subject_model", action='store_true', help="Evaluate a subject model.")
    parser.add_argument("--train_subject_models", action='store_true', help="Train the subject models.")
    parser.add_argument("--batch_size", type=int, help="Number of subject models to train. 1000 is enough in total.", default=1000)
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

    task = PermutedIrisTask(args.seed)
    epochs = 300
    trainer = AdamTrainer(task, epochs, 1000, lr=0.01, device=args.device)

    subject_model = IrisClassifier
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

        interpretability_model = Transformer(subject_model_parameter_count, task.mi_output_shape, num_layers=1, num_heads=16, positional_encoding_size=64).to(args.device)
        interpretability_model_parameter_count = sum(p.numel() for p in interpretability_model.parameters())
        print(f'Interpretability model parameter count: {interpretability_model_parameter_count}')
        train_mi_model(interpretability_model, interpretability_model_io, subject_model, subject_model_io, trainer, task, device=args.device, lr=0.001)