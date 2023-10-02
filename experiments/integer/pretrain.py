"""
Pre-train subject models for the different optimisers. This allows great
parallelism on Hamilton, rather than needing to train all the subject models as
part of the pipeline.
"""
from auto_mi.tasks import IntegerGroupFunctionRecoveryTask
from auto_mi.trainers import AdamTrainer
from auto_mi.models import IntegerGroupFunctionRecoveryModel
from auto_mi.rl import QLearner
from auto_mi.rl import pretrain_subject_models

EPISODES = 100
STEPS = 10
SUBJECT_MODEL_EPOCHS = 30
SUBJECT_MODELS_BATCH_SIZE = 2**10
SUBJECT_MODELS_PER_STEP = 10
INTERPRETABILITY_WEIGHT = 1.
DEVICE = 'cuda'
INTERPRETABILITY_BATCH_SIZE = 128
INTERPRETABILITY_MODEL_EPOCHS = 20
SUBJECT_MODEL_PATH = './subject_models'
TASK = IntegerGroupFunctionRecoveryTask(2**1 - 1, 2)
SUBJECT_MODEL = IntegerGroupFunctionRecoveryModel
HYPERPARAMETERS = {
    'weight_decay': [0, 0.1],
    'learning_rate': [0.1, 0.01],
}
STATE_SPACE = [AdamTrainer(TASK, SUBJECT_MODEL_EPOCHS, SUBJECT_MODELS_BATCH_SIZE, weight_decay=wd, lr=lr, device=DEVICE) for wd in HYPERPARAMETERS['weight_decay'] for lr in HYPERPARAMETERS['learning_rate']]
OPTIMISER_MODEL = QLearner(STATE_SPACE)

if __name__ == '__main__':
    pretrain_subject_models(OPTIMISER_MODEL, SUBJECT_MODEL_PATH, SUBJECT_MODEL, TASK)