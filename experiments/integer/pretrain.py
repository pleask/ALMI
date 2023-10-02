"""
Pre-train subject models for the different optimisers. This allows great
parallelism on Hamilton, rather than needing to train all the subject models as
part of the pipeline.
"""
from auto_mi.mi import pretrain_subject_models
from .pipeline import OPTIMISER_MODEL, SUBJECT_MODEL, SUBJECT_MODEL_PATH, TASK

if __name__ == '__main__':
    pretrain_subject_models(OPTIMISER_MODEL, SUBJECT_MODEL_PATH, SUBJECT_MODEL, TASK)