import argparse
import json
import uuid
from time import gmtime, strftime

import torch

from auto_mi.tasks import VAL

get_model_path = lambda path, net_idx: f"{path}/{net_idx}.pickle"


def assert_is_unique_model(index_file, seed, index, task, model, trainer):
    try:
        with open(index_file, 'r') as idx_file:
            for line in idx_file:
                md = json.loads(line.strip())
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
                raise ValueError(f'Model already exists as {md["id"]}')
    except FileNotFoundError:
        return


def train_subject_model_batch(task, model, trainer, seed, start_idx, end_idx, path, device='cpu'):
    for idx in range(start_idx, end_idx):
        print(f'Training model {idx} of {start_idx} to {end_idx}')
        net = model(task).to(device)
        trainer.train(net, task.get_dataset(idx))
        loss = trainer.evaluate(net, task.get_dataset(idx, type=VAL))

        assert_is_unique_model(f'{path}/index.txt', seed, idx, task, net, trainer)

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
