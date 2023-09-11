import argparse
import os
import random

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb
from auto_mi.mi import train_model
from auto_mi.mi import get_matching_subject_models_names
from auto_mi.mi import MultifunctionSubjectModelDataset
from auto_mi.tasks import IntegerGroupFunctionRecoveryTask
from auto_mi.mi import IntegerGroupFunctionRecoveryModel
from auto_mi.tasks import TASKS

os.environ["WANDB_SILENT"] = "true"

DEVICE = torch.device("cuda")
TASK = IntegerGroupFunctionRecoveryTask
BATCH_SIZE = 2**10
TRAIN_SPLIT_RATIO = 0.7
EPOCHS = 100

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
    help="Prune amount of the subject models.",
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
    wandb.config['subject_model_count'] = len(all_matching_subject_models) 

    train_sample_count = int(len(all_matching_subject_models) * TRAIN_SPLIT_RATIO)
    train_dataset = MultifunctionSubjectModelDataset(args.subject_model_dir, all_matching_subject_models[:train_sample_count])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=32)
    test_dataset = MultifunctionSubjectModelDataset(args.subject_model_dir, all_matching_subject_models[train_sample_count:])
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=32)

    model_path = args.model_path
    task = TASK
    model = IntegerGroupFunctionRecoveryModel(train_dataset.model_param_count, train_dataset.output_shape, layer_scale=10).to(DEVICE)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) #, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    print("Training model", flush=True)
    train_model(model, model_path, optimizer, EPOCHS, train_dataloader, test_dataloader, test_dataset,  criterion, task, device=DEVICE)

    losses_dict = {'op1': [], 'op2': [], 'loss': [], 'predicted_op1': [], 'predicted_op2': []}
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            outputs = model(inputs.to(DEVICE))
            targets = targets.to(DEVICE)
            l1 = F.cross_entropy(outputs[:, 0, :], targets[:, 0, :], reduction='none')
            l2 = F.cross_entropy(outputs[:, 1, :], targets[:, 1, :], reduction='none')
            losses = (l1 + l2) / 2.
            for output, target, loss in zip(outputs, targets, losses):
                d = task.decode(target)
                losses_dict['op1'].append(d[0][1])
                losses_dict['op2'].append(d[1][1])
                losses_dict['loss'].append(loss.cpu().item())
                predicted_d = task.decode(output)
                losses_dict['predicted_op1'].append(predicted_d[0][1])
                losses_dict['predicted_op2'].append(predicted_d[1][1])
    
    df = pd.DataFrame.from_dict(losses_dict)
    wandb.log({'validation_losses': wandb.Table(dataframe=df)})

    # Heatmap 
    grouped = df.groupby(['op1', 'op2'])['loss'].mean().reset_index()
    pivot_df = grouped.pivot(index='op1', columns='op2', values='loss')
    # Create the heatmap
    fig, ax = plt.subplots()

    # Plot the values using matshow
    cax = ax.matshow(pivot_df, cmap='coolwarm')

    # Add colorbar for reference
    fig.colorbar(cax)

    # Set axis labels
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))

    # Label the axis ticks
    ax.set_xticklabels(pivot_df.columns)
    ax.set_yticklabels(pivot_df.index)

    # Loop over data dimensions and create text annotations.
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            value = pivot_df.iloc[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="w")

    fig.savefig("heatmap.png")
    wandb.log({"heatmap": wandb.Image("heatmap.png")})

    samples = 10000
    random_tuples = [(random.randint(1, 1023), random.randint(1, 1023), random.randint(1, 1023)) for _ in range(samples)]
    for row in df[(df['op1'] != df['predicted_op1']) & (df['op2'] != df['predicted_op2'])].iterrows():
        correct = 0
        for t in random_tuples:
            target = eval(f'({t[0]} {row[1]["op1"]} {t[1]} {row[1]["op2"]} {t[2]}) % 8')
            output = eval(f'({t[0]} {row[1]["predicted_op1"]} {t[1]} {row[1]["predicted_op2"]} {t[2]}) % 8')
            if target == output:
                correct += 1
        print(row[1]['op1'], row[1]['op2'], row[1]['predicted_op1'], row[1]['predicted_op2'], correct / samples)
