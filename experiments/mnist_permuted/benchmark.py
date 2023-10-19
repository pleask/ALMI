import argparse
from itertools import permutations
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

from auto_mi.base import MetadataBase
from auto_mi.rl import pretrain_subject_models
from auto_mi.tasks import Task, Example, TRAIN, VAL
from auto_mi.trainers import AdamTrainer
from auto_mi.utils import TarModelWriter


TRAIN_RATIO = 0.7


class PermutedMNISTTask(Task):
    """
    The permuted MNIST interpretabilty task consists of training models to
    perform MNIST classification but with the output labels permuted (eg. 3 maps
    to 7). The interpretabilty task is figuring out what the permutation is for
    each subject model.
    """
    def __init__(self, seed=0., train=True):
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

    def get_dataset(self, i, type=TRAIN) -> Dataset:
        """
        Gets the dataset for the ith example of this task.
        """
        return PermutedMNISTExample(self._permutations[i], type=TRAIN)

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
        return F.one_hot(torch.tensor(self._permutation_map))


class MNIST_CNN(nn.Module, MetadataBase):
    def __init__(self, *_):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64*7*7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def evaluate(task, subject_model_io):
    metadata = subject_model_io.get_metadata()
    for model_idx in range(len(metadata)):
        print(f'Model {model_idx}')
        example = task.get_dataset(metadata[model_idx]['index'], type=VAL)
        model_id = metadata[model_idx]['id']
        permutation_map = metadata[model_idx]['example']['permutation_map']
        print(f"Permutation map: {permutation_map}")
        model = subject_model_io.get_model(MNIST_CNN(), model_id)
        
        for i in range(10):
            image, label = example[-i]
            image_np = image.squeeze().numpy()
            prediction = model(image.unsqueeze(0))
            print(label, torch.argmax(prediction, -1).item())
            plt.imshow(image_np, cmap='gray')
            plt.title(f"Label: {label}, Prediction: {torch.argmax(prediction, -1).item()}")
            plt.axis('off')
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run either pretraining or the full pipeline.')
    parser.add_argument("--evaluate_subject_model", action='store_true', help="Evaluate a subject model.")
    args = parser.parse_args()

    task = PermutedMNISTTask(0)
    subject_model_io = TarModelWriter('mnist/subject-models')

    if args.evaluate_subject_model:
        evaluate(task, subject_model_io)
    else:
        weight_decays = [0., 0.01, 0.001, 0.0001]
        state_space = [AdamTrainer(task, 100, 10000, weight_decay=wd, lr=0.001, device='cuda') for wd in weight_decays]
        subject_model = MNIST_CNN

        sample_model = subject_model(task)
        subject_model_parameter_count = sum(p.numel() for p in sample_model.parameters())
        print('Layer parameters')
        [print(name, p.numel()) for name, p in sample_model.named_parameters()]
        print(f'Subject model parameter count: {subject_model_parameter_count}', flush=True)

        print('Pretraining subject models')
        trainer = random.choice(state_space)
        pretrain_subject_models(trainer, subject_model_io, subject_model, task, batch_size=1)
