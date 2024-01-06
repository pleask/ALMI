from abc import ABC, abstractmethod
import argparse
import json
import os
import pickle
import random
from time import gmtime, strftime
import uuid

import filelock
import tarfile
import torch

from auto_mi.tasks import VAL
from auto_mi.mi import TRAIN_RATIO

get_model_path = lambda path, net_idx: f"{path}/{net_idx}.pickle"


def is_unique_model(index_file, seed, index, task, model, trainer):
    """
    Checks whether the model is unique (ie. whether it has been trained before
    in this experiment), and if not throws an assertion error.
    """
    with open(index_file, "r") as idx_file:
        for line in idx_file:
            # For some undetermined reason, the metadata is sometimes corrupted
            # on write and can't be read. In the mi model script, we skip over
            # these lines anyway, so here we can safely ignore that line and
            # potentially retrain the model.
            try:
                md = json.loads(line.strip())
            except json.decoder.JSONDecodeError:
                continue
            if md["task"]["seed"] != seed or md["index"] != index:
                continue
            if md["task"] != task.get_metadata():
                continue
            if md["example"] != task.get_dataset(index).get_metadata():
                continue
            if md["model"] != model.get_metadata():
                continue
            if md["trainer"] != trainer.get_metadata():
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
    def get_model(self, empty_model, id, device="cpu"):
        pass

    @abstractmethod
    def check_model_exists(self, id):
        pass


class ConcurrentMetadataWriter(ModelWriter):
    """
    Implements concurrent metadata management for an index file using a lock file.
    """

    def __init__(self, dir):
        super().__init__(dir)
        self._index_path = f"{dir}/index.txt"
        self._index_lock = f"{dir}/index.lock"
        self._metadata = None

    def write_metadata(self, md):
        lock = filelock.FileLock(self._index_lock)
        with lock, open(self._index_path, "a") as md_file:
            md_file.write(json.dumps(md) + "\n")

    def get_metadata(self):
        if self._metadata is not None:
            return self._metadata

        with open(self._index_path, "r") as file:
            metadata = []
            for line in file:
                md = json.loads(
                    line.strip().decode("utf-8")
                    if isinstance(line, bytes)
                    else line.strip()
                )
                metadata.append(md)
            self._metadata = metadata
            return metadata


class DirModelWriter(ConcurrentMetadataWriter):
    """
    Write the models to a flat hierarchy in a directory. Don't use when
    operating on more than 100k subject models if you want to tar the models, as
    this will take hours.

    It's assumed that only a single process will be writing to any model, and
    multiple processes will write to the index file.
    """

    def get_model(self, model, model_id, device="cpu"):
        model_filename = f"{model_id}.pickle"
        model_path = os.path.join(self.dir, model_filename)

        load_args = (
            (model_path,)
            if device == "cuda"
            else (model_path, {"map_location": torch.device("cpu")})
        )
        params = torch.load(*load_args)
        try:
            model.load_state_dict(params)
        except RuntimeError:
            params = {k.removeprefix("_orig_mod."): v for k,v in params.items()}
            model.load_state_dict(params)

        return model

    def write_model(self, model_id, model):
        model_path = os.path.join(self.dir, f"{model_id}.pickle")
        torch.save(model.state_dict(), model_path)

    def check_model_exists(self, model_id):
        model_path = os.path.join(self.dir, f"{model_id}.pickle")
        return os.path.exists(model_path)


class TarModelWriter(DirModelWriter):
    """
    Writes the models to one of 256 tar archives based on their ID to avoid
    write contention. Reads from a regular directory though, as untarring as a
    pre-processing step is considerably faster on the NCC.
    """

    def write_model(self, model_id, model):
        tmp_model_path = f"{self.dir}/{model_id}.pickle"
        lock_file_path = f"{self.dir}/{model_id[:2]}.lock"
        tar_archive_path = f"{self.dir}/{model_id[:2]}.tar"

        # Temporarily save the model to disk
        torch.save(
            model.state_dict(), tmp_model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL
        )

        lock = filelock.FileLock(lock_file_path)
        with lock:
            if not os.path.exists(tar_archive_path):
                # Create an empty tar file if it does not exist
                with tarfile.open(tar_archive_path, "w"):
                    pass

            with tarfile.open(tar_archive_path, "a") as tar:
                tar.add(tmp_model_path, arcname=f"{model_id}.pickle")

        os.remove(tmp_model_path)

def train_subject_models(task, model, trainer, model_writer, count=10, device="cpu"):
    """
    Trains subject models using the specified trainer. Returns the average loss
    of the subject models, and a sub-group of the trained subject models that
    are used to validate the performance of the interpretability model on
    subject models created by this trainer.
    """
    nets = [model(task).to(device) for _ in range(count)]
    targets = [task.get_dataset(i).get_target() for i in range(count)]
    losses, train_losses = trainer.train_parallel(
        nets,
        [task.get_dataset(i) for i in range(count)],
        [task.get_dataset(i, type=VAL) for i in range(count)],
    )

    model_ids = []
    for i, (net, target, loss, train_loss) in enumerate(
        zip(nets, targets, losses, train_losses)
    ):
        net_id = str(uuid.uuid4())
        model_writer.write_metadata(
            md={
                "task": task.get_metadata(),
                "example": task.get_dataset(i).get_metadata(),
                "model": net.get_metadata(),
                "trainer": trainer.get_metadata(),
                "loss": loss,
                "train_loss": train_loss,
                "id": str(net_id),
                "time": strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                "index": i,
            }
        )
        model_writer.write_model(net_id, net)

        model_ids.append(str(net_id))

    subject_model_loss = sum(losses) / len(losses)

    validation_subject_model_ids = model_ids[int(TRAIN_RATIO * count) :]

    return subject_model_loss, validation_subject_model_ids


def get_args_for_slum():
    """
    Sets up and gets the args necessary for running this in parallel on slurm.
    """
    parser = argparse.ArgumentParser(
        description="Trains subject models, ie. the models that implement the labeling function."
    )
    parser.add_argument(
        "--path", type=str, help="Directory to which to save the models"
    )
    args = parser.parse_args()
    return args


def evaluate_subject_model(
    task_class, subject_model_class, subject_model_io, samples=100, model_count=100
):
    metadata = subject_model_io.get_metadata()
    accuracies = []
    for model_idx in range(min(model_count, len(metadata))):
        print(f"Model {model_idx}")
        task = task_class(seed=metadata[model_idx]["task"]["seed"])
        example = task.get_dataset(metadata[model_idx]["index"], type=VAL)
        model_id = metadata[model_idx]["id"]
        permutation_map = metadata[model_idx]["example"]["permutation_map"]
        print(f"Permutation map: {permutation_map}")
        model = subject_model_io.get_model(subject_model_class(), model_id)
        correct = []
        for _ in range(samples):
            i = random.randint(0, len(example) - 1)
            input, label = example[i]
            prediction = model(torch.Tensor(input).unsqueeze(0))
            correct.append((torch.argmax(prediction, -1) == label)[0].item())
        accuracy = sum(correct) / samples
        print(accuracy)
        accuracies.append(accuracy)
    print("Overall accuracy:", sum(accuracies) / len(accuracies))