"""
Evaluate pruned versions of the networks.
"""
import argparse
import json

import torch
from torch.nn.utils.prune import L1Unstructured
from torch import nn

from train_subject_models import get_subject_fn, evaluate_subject_nets, get_model_path, get_metadata_path
from train_multifunction_mi_model import get_matching_subject_models_names, get_subject_model, get_subject_model_metadata


def prune(model, amount=0.2):
    pruner = L1Unstructured(amount)
    layers = list(model.children())

    prune_parameters = []
    for layer in layers:
        if isinstance(layer, nn.Linear):
            for name, param in layer.named_parameters():
                prune_parameters.append((layer, name))

    # This needs to happen in a separate loop so we don' modify the parameter
    # dictionary whilst we're iterarting over it.
    for (layer, name) in prune_parameters:
        pruner.apply(layer, name, amount=amount)
        
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--subject_model_dir", help="Folder containing the subject models")
parser.add_argument("--pruned_model_losses", help="File to save losses")
parser.add_argument("--batch_size", default=100, help="Number of models to evaluate at a time")
parser.add_argument("--amount", default=0.2, help="Amount of parameters to prune.")

if __name__ == '__main__':
    args = parser.parse_args()

    model_names = get_matching_subject_models_names(args.subject_model_dir)
    print(f'Found {len(model_names)} models')

    for i in range(0, len(model_names), args.batch_size):
        names = model_names[i:i+args.batch_size]

        models, fns, metadatas = [], [], []
        for name in names:
            model = get_subject_model(args.subject_model_dir, name, device='cpu')
            models.append(prune(model, amount=args.amount))
            metadata = get_subject_model_metadata(args.subject_model_dir, name)
            metadatas.append(metadata)
            fns.append(get_subject_fn(metadata['fn_name'], metadata['parameter']))

        losses = evaluate_subject_nets(models, fns)
        
        with open(args.pruned_model_losses, 'a') as o:
            for name, loss in zip(names, losses):
                o.write(f'{name} {loss}\n')
