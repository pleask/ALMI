from abc import ABC, abstractmethod
import filelock
import json
import os
import pickle

import torch


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