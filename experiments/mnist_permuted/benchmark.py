import argparse
from itertools import permutations
import random

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
from auto_mi.utils import DirModelWriter, TarModelWriter


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
            print(label, torch.argmax(prediction, -1).item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run either pretraining or the full pipeline.')
    parser.add_argument("--evaluate_subject_model", action='store_true', help="Evaluate a subject model.")
    parser.add_argument("--seed", type=float, help="Random seed.", default=0.)
    parser.add_argument("--device", type=str, default="cuda", help="Device to train or evaluate models on")
    parser.add_argument("--subject_model_path", type=str, help="Path of the subject models")
    args = parser.parse_args()

    subject_model_io = TarModelWriter(args.subject_model_path)

    if args.evaluate_subject_model:
        evaluate(subject_model_io)
    else:
        task = PermutedMNISTTask(args.seed)

        base_net = MNIST_CNN()
        model_path = f'experiments/mnist_permuted/base_net.pickle'
        load_args = (model_path,) if args.device == 'cuda' else (model_path, {'map_location': torch.device('cpu')})
        try:
            base_net.load_state_dict(torch.load(*load_args))
        except FileNotFoundError:
            pass

        l1_penalty_weights = [0.]
        epochs = 30
        state_space = [AdamTrainer(task, epochs, 1000, lr=0.01, l1_penalty_weight=l1pw, device=args.device) for l1pw in l1_penalty_weights]
        subject_model = MNIST_CNN

        sample_model = subject_model(task)
        subject_model_parameter_count = sum(p.numel() for p in sample_model.parameters())
        print('Layer parameters')
        [print(name, p.numel()) for name, p in sample_model.named_parameters()]
        print(f'Subject model parameter count: {subject_model_parameter_count}', flush=True)

        print('Pretraining subject models')
        trainer = random.choice(state_space)
        pretrain_subject_models(trainer, subject_model_io, subject_model, task, batch_size=5)
