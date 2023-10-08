from abc import ABC, abstractmethod
import argparse
import json
import os
from time import gmtime, strftime
import uuid

import filelock
import tarfile
import tempfile
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


class ModelWriter(ABC):
    def __init__(self, dir):
        self.dir = dir

    @abstractmethod
    def write_metadata(self, md):
        pass

    @abstractmethod
    def write_model(self, id, model):
        pass

    @abstractmethod
    def get_metadata(self):
        pass

    @abstractmethod
    def get_model(self, empty_model, id, device='cpu'):
        pass


class TarModelWriter(ModelWriter):
    def __init__(self, dir: str):
        super().__init__(dir)
        self._index_path = f'{dir}/index.txt'
        self._index_lock = f'{dir}/index.lock'

    def write_metadata(self, md):
        print('Acquiring lock to write metadata')
        lock = filelock.FileLock(self._index_lock)
        with lock, open(self._index_path, 'a') as md_file:
            md_file.write(json.dumps(md) + '\n')

    def write_model(self, model_id, model):
        tmp_model_path = f'{self.dir}/{model_id}.pickle'
        lock_file_path = f'{self.dir}/{model_id[:2]}.lock'
        tar_archive_path = f'{self.dir}/{model_id[:2]}.tar'

        # Temporarily save the model to disk
        torch.save(model.state_dict(), tmp_model_path)

        print(f'Acquiring lock to write model {model_id}')
        lock = filelock.FileLock(lock_file_path)
        with lock:
            if not os.path.exists(tar_archive_path):
                # Create an empty tar file if it does not exist
                with tarfile.open(tar_archive_path, 'w'):
                    pass

            with tarfile.open(tar_archive_path, 'a') as tar:
                tar.add(tmp_model_path, arcname=f'{model_id}.pickle')

        os.remove(tmp_model_path)

    def get_metadata(self):
        with open(self._index_path, 'r') as file:
            metadata = []
            for line in file:
                md = json.loads(line.strip().decode('utf-8') if isinstance(line, bytes) else line.strip())
                metadata.append(md)
            return metadata

    def get_model(self, model, model_id, device='cpu'):
        model_filename = f'{model_id}.pickle'
        extracted_model_path = os.path.join(self.dir, model_filename)

        if not os.path.exists(extracted_model_path):
                tar_archive_path = os.path.join(self.dir, f'{model_id[:2]}.tar')
                with tarfile.open(tar_archive_path, 'r') as tar:
                    tar.extract(model_filename, path=self.dir)

        load_args = (extracted_model_path,) if device == 'cuda' else (extracted_model_path, {'map_location': torch.device('cpu')})
        model.load_state_dict(torch.load(*load_args))

        return model



def train_subject_models(task, model, trainer, model_writer, count=10, device='cpu'):
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
        net_id = str(uuid.uuid4())
        model_writer.write_metadata(
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
        )
        model_writer.write_model(net_id, net)

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
