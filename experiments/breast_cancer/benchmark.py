import argparse
import random

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import wandb

from auto_mi.base import MetadataBase
from auto_mi.mi import PositionalEncoding, Transformer, train_mi_model
from auto_mi.rl import pretrain_subject_models
from auto_mi.sklearn import SklearnExample, SklearnTask
from auto_mi.tasks import TRAIN, VAL
from auto_mi.trainers import AdamTrainer
from auto_mi.utils import DirModelWriter, TarModelWriter, evaluate_subject_model

class PermutedBreastCancerTask(SklearnTask):
    def __init__(self, seed=0., train=True, **kwargs):
        super().__init__(PermutedBreastCancerExample, (30,), 2, seed=seed, train=train)


class PermutedBreastCancerExample(SklearnExample):
    def __init__(self, permutation_map, type=TRAIN):
        super().__init__(permutation_map, type)
    
    def _get_dataset(self):
        dataset = datasets.load_breast_cancer()
        X = dataset.data.astype(float)
        y = dataset.target

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y


class BreastCancerClassifier(nn.Module, MetadataBase):
    def __init__(self, *_):
        super().__init__()
        self.fc1 = nn.Linear(30, 30)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(30, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


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

    task = PermutedBreastCancerTask(args.seed)

    if args.evaluate_subject_model:
        evaluate_subject_model(PermutedBreastCancerTask, BreastCancerClassifier, subject_model_io)
        quit()

    epochs = 50
    trainer = AdamTrainer(task, epochs, 1000, lr=0.01, device=args.device)

    subject_model = BreastCancerClassifier
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

        positional_encoding = PositionalEncoding(8, subject_model_parameter_count)
        interpretability_model = Transformer(subject_model_parameter_count, task.mi_output_shape, positional_encoding, num_layers=12).to(args.device)
        interpretability_model_parameter_count = sum(p.numel() for p in interpretability_model.parameters())
        print(f'Interpretability model parameter count: {interpretability_model_parameter_count}')
        trainer = AdamTrainer(task, 100, 1000, lr=0.01, device=args.device)
        train_mi_model(interpretability_model, interpretability_model_io, subject_model, subject_model_io, trainer, task, device=args.device)