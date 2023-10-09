"""
Implements the full meta-learning pipeline.
"""
import argparse
import os

import torch
import wandb

from auto_mi.tasks import IntegerGroupFunctionRecoveryTask
from auto_mi.trainers import AdamTrainer
from auto_mi.mi import Transformer, FeedForwardNN
from auto_mi.models import IntegerGroupFunctionRecoveryModel
from auto_mi.rl import QLearner, train_optimiser_model, pretrain_subject_models
from auto_mi.utils import TarModelWriter

EPISODES = 100
STEPS = 10
SUBJECT_MODEL_EPOCHS = 100
SUBJECT_MODELS_BATCH_SIZE = 2**10
SUBJECT_MODELS_PER_STEP = 10
INTERPRETABILITY_WEIGHT = 1.
DEVICE = 'cpu'
INTERPRETABILITY_BATCH_SIZE = 128
INTERPRETABILITY_MODEL_EPOCHS = 5
TASK = IntegerGroupFunctionRecoveryTask(2**1 - 1, 2)
SUBJECT_MODEL = IntegerGroupFunctionRecoveryModel
HYPERPARAMETERS = {
    'weight_decay': [0, 0.001, 0.01, 0.1],
    'learning_rate': [0.1, 0.01, 0.001, 0.0001],
}
STATE_SPACE = [AdamTrainer(TASK, SUBJECT_MODEL_EPOCHS, SUBJECT_MODELS_BATCH_SIZE, weight_decay=wd, lr=lr, device=DEVICE) for wd in HYPERPARAMETERS['weight_decay'] for lr in HYPERPARAMETERS['learning_rate']]
OPTIMISER_MODEL = QLearner

# Set up the interpretability model
# TODO: Having to initialise a sample model is not lovely
sample_model = SUBJECT_MODEL(TASK)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run either pretraining or the full pipeline.')
    parser.add_argument("--pretrain", action="store_true", help="Pretrain models rather than running the full pipeline")
    parser.add_argument("--array_number", type=int, default=-1, help="Test a single hyperparameter combination rather than running the full pipeline. This facilitates parallelisation.")
    parser.add_argument("--subject_model_path", type=str, help="Location of the subject models. If writing to this location (ie. when pretraining or running the RL pipeline with training new subject models) it must be a directory. If only reading from this location (ie. when running the pipeline without training new subject models), can be a tar archive.")
    args = parser.parse_args()

    model_writer = TarModelWriter(args.subject_model_path)

    if args.pretrain:
        print('Pretraining subject models')
        optimiser_model = OPTIMISER_MODEL(STATE_SPACE)
        pretrain_subject_models(optimiser_model, model_writer, SUBJECT_MODEL, TASK, batch_size=5)
    else:
        print('Running RL pipeline')
        torch.multiprocessing.set_start_method('spawn')
        subject_model_parameter_count = sum(p.numel() for p in sample_model.parameters())
        print(f'Subject model parameter count: {subject_model_parameter_count}')

        state_space = STATE_SPACE
        name = 'Full pipeline run'
        if args.array_number >= 0:
            state_space = [STATE_SPACE[args.array_number]]
            name = f'Array run {args.array_number}: {state_space[0].get_metadata()}'

        optimiser_model = OPTIMISER_MODEL(state_space)

        interpretability_models = [FeedForwardNN(subject_model_parameter_count, TASK.mi_output_shape).to(DEVICE) for _ in state_space]

        # TODO: Use an evaluator structure instead of function on task
        os.environ["WAND_API_KEY"] = "dd685d7aa9b38a2289e5784a961b81e22fc4c735"
        wandb.init(project='bounding-mi', entity='patrickaaleask', reinit=True, name=name)

        train_optimiser_model(optimiser_model, interpretability_models, model_writer, SUBJECT_MODEL, TASK, EPISODES, STEPS, subject_models_per_step=SUBJECT_MODELS_PER_STEP)

        # Return the best hyperparameters
        print(f"Optimal weight decay is {OPTIMISER_MODEL.get_optimal().weight_decay}, optimal lr is {OPTIMISER_MODEL.get_optimal().lr}")
