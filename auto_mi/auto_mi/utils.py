import argparse
import json
import uuid
from time import gmtime, strftime

import torch

from auto_mi.tasks import VAL
from auto_mi.mi import TRAIN_RATIO

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


def train_subject_models(task, model, trainer, subject_model_path, count=10, device='cpu'):
    """
    Trains subject models using the specified trainer. Returns the average loss
    of the subject models, and a sub-group of the trained subject models that
    are used to validate the performance of the interpretability model on
    subject models created by this trainer.
    """
    nets = [model(task).to(device) for _ in range(count)]
    targets = [task.get_dataset(i).get_target() for i in range(count)]
    losses = trainer.train_parallel(
        nets,
        [task.get_dataset(i) for i in range(count)],
        [task.get_dataset(i, type=VAL) for i in range(count)],
    )

    model_ids = []
    for i, (net, target, loss) in enumerate(zip(nets, targets, losses)):
        net_id = uuid.uuid4()
        model_path = f'{subject_model_path}/{net_id}.pickle'
        torch.save(net.state_dict(), model_path)
        md = {
            'task': task.get_metadata(),
            'example': task.get_dataset(i).get_metadata(),
            'model': net.get_metadata(),
            'trainer': trainer.get_metadata(),
            "loss": loss,
            "id": str(net_id),
            "time": strftime("%Y-%m-%d %H:%M:%S", gmtime()),
            "index": i,
        }
        with open(f'{subject_model_path}/index.txt', 'a') as md_file:
            md_file.write(json.dumps(md) + '\n')

        model_ids.append(str(net_id))
    
    subject_model_loss = sum(losses) / len(losses)

    validation_subject_model_ids = model_ids[int(TRAIN_RATIO * count):]

    return subject_model_loss, validation_subject_model_ids


def get_args_for_slum():
    """
    Sets up and gets the args necessary for running this in parallel on slurm.
    """
    parser = argparse.ArgumentParser(
        description="Trains subject models, ie. the models that implement the labeling function."
    )
    parser.add_argument("--path", type=str, help="Directory to which to save the models")
    args = parser.parse_args()
    return args
