"""
Implements the full meta-learning pipeline.
"""
import os

import wandb

from auto_mi.tasks import IntegerGroupFunctionRecoveryTask
from auto_mi.trainers import AdamTrainer
from auto_mi.mi import Transformer
from auto_mi.models import IntegerGroupFunctionRecoveryModel
from auto_mi.rl import QLearner, train_optimiser_model

EPISODES = 100
STEPS = 10
SUBJECT_MODEL_EPOCHS = 30
SUBJECT_MODELS_BATCH_SIZE = 2**10
SUBJECT_MODELS_PER_STEP = 10
INTERPRETABILITY_WEIGHT = 1.
DEVICE = 'cpu'
INTERPRETABILITY_BATCH_SIZE = 128
INTERPRETABILITY_MODEL_EPOCHS = 20
SUBJECT_MODEL_PATH = './subject_models'

task = IntegerGroupFunctionRecoveryTask(2**1 - 1, 2)

hyperparameters = {
    'weight_decay': [0, 0.1],
    'learning_rate': [0.1, 0.01],
}
state_space = [AdamTrainer(task, SUBJECT_MODEL_EPOCHS, SUBJECT_MODELS_BATCH_SIZE, weight_decay=wd, lr=lr, device=DEVICE) for wd in hyperparameters['weight_decay'] for lr in hyperparameters['learning_rate']]

optimiser_model = QLearner(state_space)

# Set up the interpretability model
# TODO: Having to initialise a sample model is not lovely
sample_model = IntegerGroupFunctionRecoveryModel(task)
subject_model_parameter_count = sum(p.numel() for p in sample_model.parameters())
print(f'Subject model parameter count: {subject_model_parameter_count}')
interpretability_model = Transformer(subject_model_parameter_count, task.mi_output_shape).to(DEVICE)

# TODO: Use an evaluator structure instead of function on task
os.environ["WAND_API_KEY"] = "dd685d7aa9b38a2289e5784a961b81e22fc4c735"
wandb.init(project='bounding-mi', entity='patrickaaleask', reinit=True)

train_optimiser_model(optimiser_model, interpretability_model, SUBJECT_MODEL_PATH, IntegerGroupFunctionRecoveryModel, task, EPISODES, STEPS, subject_models_per_step=SUBJECT_MODELS_PER_STEP)

# Return the best hyperparameters
print(f"Optimal weight decay is {optimiser_model.get_optimal().weight_decay}, optimal lr is {optimiser_model.get_optimal().lr}")