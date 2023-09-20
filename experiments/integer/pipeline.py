"""
Implements the full meta-learning pipeline.
"""
import os

import matplotlib.pyplot as plt
import wandb

from auto_mi.tasks import IntegerGroupFunctionRecoveryTask, VAL
from auto_mi.base import MetadataBase
from auto_mi.trainers import AdamTrainer
from auto_mi.mi import Transformer, MultifunctionSubjectModelDataset, train_interpretability_model
from auto_mi.models import IntegerGroupFunctionRecoveryModel
from auto_mi.rl import QLearner, train_optimiser_model

EPISODES = 1
STEPS = 1
SUBJECT_MODEL_EPOCHS = 3
SUBJECT_MODELS_BATCH_SIZE = 2**10
SUBJECT_MODELS_PER_STEP = 1
INTERPRETABILITY_WEIGHT = 1.
DEVICE = 'cpu'
INTERPRETABILITY_BATCH_SIZE = 128
INTERPRETABILITY_MODEL_EPOCHS = 2
SUBJECT_MODEL_PATH = './subject_models'

task = IntegerGroupFunctionRecoveryTask(2**3 - 1, 2)

hyperparameters = {
    'weight_decay': [0, 0.1],
    'learning_rate': [0.1, 0.01],
}
state_space = [AdamTrainer(task, SUBJECT_MODEL_EPOCHS, SUBJECT_MODELS_BATCH_SIZE, weight_decay=wd, lr=lr, device=DEVICE) for wd in hyperparameters['weight_decay'] for lr in hyperparameters['learning_rate']]

optimiser_model = QLearner(state_space)

# Set up the interpretability model
# TODO: Get these sizes from the task 
interpretability_model = Transformer(213, (1, 2)).to(DEVICE)

# TODO: Use an evaluator structure instead of function on task
os.environ["WAND_API_KEY"] = "dd685d7aa9b38a2289e5784a961b81e22fc4c735"
wandb.init(project='bounding-mi', entity='patrickaaleask', reinit=True)

train_optimiser_model(optimiser_model, interpretability_model, SUBJECT_MODEL_PATH, IntegerGroupFunctionRecoveryModel, task, EPISODES, STEPS)

# Return the best hyperparameters
print(f"Optimal weight decay is {optimiser_model.get_optimal().weight_decay}, optimal lr is {optimiser_model.get_optimal().lr}")