import argparse
from itertools import permutations
import os
import random

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


class PermutedMNISTTask(Task):
    """
    The permuted MNIST interpretabilty task consists of training models to
    perform MNIST classification but with the output labels permuted (eg. 3 maps
    to 7). The interpretabilty task is figuring out what the permutation is for
    each subject model.
    """
    def __init__(self, seed=0., train=True, **kwargs):
        super().__init__(seed=seed, train=train)
        p = list(permutations(range(10)))
        # Shuffle the permutations so we see examples where all output classes
        # are remapped.
        r = random.Random(seed)
        r.shuffle(p)
        if type == TRAIN:
            self._permutations = p[:int(TRAIN_RATIO * len(p))]
        else:
            self._permutations = p[int(TRAIN_RATIO * len(p)):]

    def get_dataset(self, i, type=TRAIN, **_) -> Dataset:
        """
        Gets the dataset for the ith example of this task.
        """
        return PermutedMNISTExample(self._permutations[i], type=type)

    @property
    def input_shape(self):
        return (28, 28)

    @property
    def output_shape(self):
        return (10, )

    @property
    def mi_output_shape(self):
        return (10, 10)

    def criterion(self, x, y):
        return F.nll_loss(x, y)


class PermutedMNISTExample(Example):
    def __init__(self, permutation_map, type=TRAIN):
        self._permutation_map = permutation_map
        if type == TRAIN:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            self._dataset = torchvision.datasets.MNIST(root='./data', train=type==TRAIN, transform=transform, download=True)

    def __getitem__(self, i):
        img, target = self._dataset[i]
        return img, self._permutation_map[target]

    def __len__(self):
        return len(self._dataset)

    def get_metadata(self):
        return {'permutation_map': self._permutation_map}
    
    def get_target(self):
        return F.one_hot(torch.tensor(self._permutation_map)).to(torch.float32)


class MNIST_CNN(nn.Module, MetadataBase):
    def __init__(self, *_):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(4*7*7, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 8)
        self.fc4 = nn.Linear(8, 8)
        self.fc5 = nn.Linear(8, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 4*7*7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


def evaluate(subject_model_io):
    metadata = subject_model_io.get_metadata()
    for model_idx in range(len(metadata)):
        print(f'Model {model_idx}')
        task = PermutedMNISTTask(seed=metadata[model_idx]['task']['seed'])
        example = task.get_dataset(metadata[model_idx]['index'], type=VAL)
        model_id = metadata[model_idx]['id']
        permutation_map = metadata[model_idx]['example']['permutation_map']
        print(f"Permutation map: {permutation_map}")
        model = subject_model_io.get_model(MNIST_CNN(), model_id)
        
        for i in range(10):
            image, label = example[-i]
            prediction = model(image.unsqueeze(0))
            print(prediction)
            print(label, torch.argmax(prediction, -1).item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run either pretraining or the full pipeline.')
    parser.add_argument("--evaluate_subject_model", action='store_true', help="Evaluate a subject model.")
    parser.add_argument("--train_subject_models", action='store_true', help="Train the subject models.")
    parser.add_argument("--seed", type=float, help="Random seed.", default=0.)
    parser.add_argument("--device", type=str, default="cuda", help="Device to train or evaluate models on")
    parser.add_argument("--subject_model_path", type=str, help="Path of the subject models")
    parser.add_argument("--interpretability_model_path", type=str, help="Path of the interpretability models")
    args = parser.parse_args()

    subject_model_io = TarModelWriter(args.subject_model_path)
    interpretability_model_io = TarModelWriter(args.interpretability_model_path)

    if args.evaluate_subject_model:
        evaluate(subject_model_io)
        quit()

    task = PermutedMNISTTask(args.seed)
    epochs = 30
    trainer = AdamTrainer(task, epochs, 1000, lr=0.01, l1_penalty_weight=0., device=args.device)

    subject_model = MNIST_CNN
    sample_model = subject_model(task)
    subject_model_parameter_count = sum(p.numel() for p in sample_model.parameters())
    print('Layer parameters')
    print(f'Subject model parameter count: {subject_model_parameter_count}', flush=True)

    if args.train_subject_models:
        base_net = MNIST_CNN()
        model_path = f'experiments/mnist_permuted/base_net.pickle'
        load_args = (model_path,) if args.device == 'cuda' else (model_path, {'map_location': torch.device('cpu')})
        try:
            base_net.load_state_dict(torch.load(*load_args))
        except FileNotFoundError:
            pass

        l1_penalty_weights = [0.]
        state_space = [trainer]


        print('Pretraining subject models')
        trainer = random.choice(state_space)
        pretrain_subject_models(trainer, subject_model_io, subject_model, task, batch_size=5)
    else:
        wandb.init(project='bounding-mi', entity='patrickaaleask', reinit=True)

        interpretability_model = Transformer(subject_model_parameter_count, task.mi_output_shape, hidden_size=256, num_layers=6, num_heads=8).to(args.device)
        interpretability_model_parameter_count = sum(p.numel() for p in interpretability_model.parameters())
        print(f'Interpretability model parameter count: {interpretability_model_parameter_count}')
        train_mi_model(interpretability_model, interpretability_model_io, subject_model, subject_model_io, trainer, task, device=args.device)