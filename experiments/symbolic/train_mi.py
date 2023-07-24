import argparse
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from auto_mi.mi import train_model
from auto_mi.mi import evaluate_model
from auto_mi.mi import get_matching_subject_models_names
from auto_mi.mi import MultifunctionSubjectModelDataset

from auto_mi.models import FeedForwardNN2D
from auto_mi.tasks import TASKS

os.environ["WANDB_SILENT"] = "true"

DEVICE = torch.device("cuda")
TASK = 'SymbolicFunctionRecoveryTask'
BATCH_SIZE = 1024
TRAIN_SPLIT_RATIO = 0.7

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="Repeat number")
parser.add_argument("--subject_model_dir", help="Folder containing the subject models")
parser.add_argument("--model_path", type=str, help="Path to save this model")
parser.add_argument(
    "--weight_decay",
    type=float,
    help="Weight decay for subject models",
    default=0.
)
parser.add_argument(
    "--prune_amount",
    type=float,
    help="Amount by which to prune the subject models before training on them.",
    default=0.
)

if __name__ == '__main__':
    os.environ["WAND_API_KEY"] = "dd685d7aa9b38a2289e5784a961b81e22fc4c735"

    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')

    random.seed(a=args.seed)
    wandb.init(config=args, project='bounding-mi', entity='patrickaaleask', reinit=True)

    all_matching_subject_models = get_matching_subject_models_names(args.subject_model_dir, task=TASK, max_loss=1., weight_decay=args.weight_decay, prune_amount=args.prune_amount)
    print(f"Found {len(all_matching_subject_models)}", flush=True)
    if len(all_matching_subject_models) == 0:
        quit()
    wandb.config['subject_model_count'] = len(all_matching_subject_models) 

    train_sample_count = int(len(all_matching_subject_models) * TRAIN_SPLIT_RATIO)
    train_dataset = MultifunctionSubjectModelDataset(args.subject_model_dir, all_matching_subject_models[:train_sample_count])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = MultifunctionSubjectModelDataset(args.subject_model_dir, all_matching_subject_models[train_sample_count:], prune_amount=args.prune_amount)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model_path = args.model_path
    task = TASKS[TASK]
    model = FeedForwardNN2D(train_dataset.model_param_count, train_dataset.output_shape, layer_scale=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.MSELoss()

    print("Training model", flush=True)
    train_model(model, model_path, optimizer, 1, train_dataloader, test_dataloader, test_dataset,  criterion, device=DEVICE)

    print("Prediction sample", flush=True)
    evaluate_model(model, test_dataloader, device=DEVICE)
