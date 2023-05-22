"""
Trains a network to recover both the function that the network implements, and
the coefficient.
"""
import argparse
import random

import torch
from torch.utils.data import Dataset, DataLoader

from .train_mi_model import get_subject_net, load_subject_model, SubjectModelDataset, DEVICE, MI_MODEL_TRAIN_SPLIT_RATIO, SUBJECT_MODEL_PARAMETER_COUNT, MI_CRITERION, SUBJECT_LAYER_SIZE, get_subject_model_dataloaders, Transformer, train_model

random.seed(a=0)

class MultifunctionSubjectModelDataset(Dataset):
    """
    Wraps function datasets (ie. a dataset of subject models that implement
    addition) to train a model to identify both the function and the
    coefficient.
    """
    def __init__(self, addition, multiplication, sigmoid, exponent, min):
        self._keys = ['addition', 'multiplication', 'sigmoid', 'exponent', 'min']
        self._function_datasets = {
            self._keys[0]: addition,
            self._keys[1]: multiplication,
            self._keys[2]: sigmoid,
            self._keys[3]: exponent,
            self._keys[4]: min,
        }

    @property
    def fn_count(self):
        return len(self._keys)

    def get_dataset_index(self, i):
        fn_name = self._keys[i % self.fn_count]
        fn_dataset = self._function_datasets[fn_name]
        return fn_dataset[i // self.fn_count]

    def __len__(self):
        return sum(len(v) for v in self._function_datasets.values())

    def __getitem__(self, idx):
        function_name = self._keys[idx % self.fn_count]
        function_idx = idx // self.fn_count
        x, coef = self._function_datasets[function_name][function_idx]
        one_hot = [1. if i == function_idx else 0. for i in range(self.fn_count)]
        one_hot.append(coef)
        return x, one_hot


parser = argparse.ArgumentParser()
parser.add_argument('--model_folders', nargs='+', help='Folders containing the subject models for each function in the order addition, multiplication, sigmoid, exponent, min')
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    help="Number of epochs for which to train.",
)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    help="Optionally load a model to evaluate or continue training.",
)

if __name__ == '__main__':
    args = parser.parse_args()
    
    print("CREATING DATALOADERS")
    function_dataloaders = [get_subject_model_dataloaders(subject_dir) for subject_dir in args.model_folders]
    train_dataset = MultifunctionSubjectModelDataset(dl[0] for dl in function_dataloaders)
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_dataset = MultifunctionSubjectModelDataset(dl[1] for dl in function_dataloaders)
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=True)

    print("CREATING MODEL")
    model = Transformer(SUBJECT_MODEL_PARAMETER_COUNT, 6, num_heads=6, hidden_size=240)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    print("TRAINING MODEL")
    train_model(model, args.model_path, optimizer, train_dataloader, test_dataloader, test_dataset)