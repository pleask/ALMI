"""
Evaluate pruned versions of the networks.
"""
import argparse
import json
import torch

from .train_subject_models import get_subject_fn, evaluate_subject_nets, get_model_path, get_metadata_path
from .train_multifunction_mi_model import get_matching_subject_models_names, get_subject_model, get_subject_model_metadata

def prune(model):
    return None

parser = argparse.ArgumentParser()
parser.add_argument("--subject_model_dir", help="Folder containing the subject models")
parser.add_argument("--pruned_model_dir", help="Folder to save the pruned subject models")
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
            model = get_subject_model(name)
            pruned_model = prune(model, amount=args.amount)

            metadata = get_subject_model_metadata(name)
            fn = get_subject_fn(metadata['fn_name'], metadata['parameter'])

        losses = evaluate_subject_nets(models, fns)

        for i, loss in losses:
            metadata[i]['loss'] = loss

        for name, model, metadata in zip(names, models, metadatas):
            model_path = get_model_path(args.pruned_model_dir, name)
            torch.save(model.state_dict(), model_path)
            metadata_path = get_metadata_path(args.pruned_model_dir, name)
            with open(metadata_path, "w") as json_file:
                json.dump(metadata, json_file)

