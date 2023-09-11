import argparse
import json
import uuid
from time import gmtime, strftime
import random

import torch

from auto_mi.tasks import VAL

get_model_path = lambda path, net_idx: f"{path}/{net_idx}.pickle"


def is_unique_model(index_file, seed, index, task, model, trainer):
    """
    Checks whether the model is unique (ie. whether it has been trained before
    in this experiment), and if not throws an assertion error.
    """
    with open(index_file, 'r') as idx_file:
        for line in idx_file:
            # For some undetermined reason, the metadata is sometimes corrupted
            # on write and can't be read. In the mi model script, we skip over
            # these lines anyway, so here we can safely ignore that line and
            # potentially retrain the model.
            try:
                md = json.loads(line.strip())
            except json.decoder.JSONDecodeError:
                continue
            if md['task']['seed'] != seed or md['index'] != index:
                continue
            if md['task'] != task.get_metadata():
                continue
            if md['example'] != task.get_dataset(index).get_metadata():
                continue
            if md['model'] != model.get_metadata():
                continue
            if md['trainer'] != trainer.get_metadata():
                continue
            return False

    return True


def train_subject_model_batch(task, model, trainer, seed, start_idx, end_idx, path, device='cpu'):
    for idx in range(start_idx, end_idx):
        random.seed(a=idx)
        net = model(task).to(device)

        print(sum(p.numel() for p in net.parameters()))

        try:
            assert is_unique_model(f'{path}/index.txt', seed, idx, task, net, trainer)
        except FileNotFoundError:
            pass
        except AssertionError:
            print(f'Skipping model {idx} of {start_idx} to {end_idx}', flush=True)
            continue

        print(f'Training model {idx} of {start_idx} to {end_idx}', flush=True)
        loss = trainer.train(net, task.get_dataset(idx), task.get_dataset(idx, type=VAL))
        print(f'Validation loss: {loss}')

        net_id = uuid.uuid4()
        md = {
            'task': task.get_metadata(),
            'example': task.get_dataset(idx).get_metadata(),
            'model': net.get_metadata(),
            'trainer': trainer.get_metadata(),
            "loss": loss,
            "id": str(net_id),
            "time": strftime("%Y-%m-%d %H:%M:%S", gmtime()),
            "index": idx,
        }

        with open(f'{path}/index.txt', 'a') as md_file:
            md_file.write(json.dumps(md) + '\n')

        model_path = get_model_path(path, net_id) 
        torch.save(net.state_dict(), model_path)

def get_args_for_slum():
    """
    Sets up and gets the args necessary for running this in parallel on slurm.
    """
    parser = argparse.ArgumentParser(
        description="Trains subject models, ie. the models that implement the labeling function."
    )
    parser.add_argument("--path", type=str, help="Directory to which to save the models")
    parser.add_argument(
        "--seed",
        type=int,
        help="The random seed to use at an experiment level. Used in conjunction with the index to provide a random seed to each model training run.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        help="The start index of the subject models.",
        required=True
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        help="The end index of the subject models (non-inclusive).",
        required=True
    )
    args = parser.parse_args()
    return args
