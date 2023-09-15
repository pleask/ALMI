"""
Implements the full meta-learning pipeline.
"""
import json
import os
import math
import numpy as np
from time import strftime, gmtime
import uuid

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import matplotlib.pyplot as plt
import wandb

from auto_mi.tasks import IntegerGroupFunctionRecoveryTask, VAL
from auto_mi.base import MetadataBase
from auto_mi.trainers import AdamTrainer
from auto_mi.mi import Transformer, get_matching_subject_models_names, MultifunctionSubjectModelDataset
from auto_mi.models import IntegerGroupFunctionRecoveryModel

EPISODES = 5
STEPS = 5
SUBJECT_MODEL_EPOCHS = 10
SUBJECT_MODEL_BATCH_SIZE = 2**10
SUBJECT_MODELS_PER_STEP = 3
INTERPRETABILITY_WEIGHT = 0.5
DEVICE = 'cpu'
INTERPRETABILITY_BATCH_SIZE = 128
INTERPRETABILITY_MODEL_EPOCHS = 20
SUBJECT_MODEL_PATH = './subject_models'

# Define the subject model task we're trying to solve here
task = IntegerGroupFunctionRecoveryTask(2**3 - 1, 3)

# Define the hyperparameter search space
hyperparameters = {
    'weight_decay': [0, 0.1],
    'learning_rate': [0.1, 0.01],
}
state_space = [(wd, lr) for wd in hyperparameters['weight_decay'] for lr in hyperparameters['learning_rate']]

# Set up the reinforcement learning algorithm for the optimiser
class RL:
    def __init__(self, state_space, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.state_space = state_space
        self.q_table = np.zeros((len(state_space), len(state_space)))

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.state_space))
        else:
            return np.argmax(self.q_table[state, :])

    def update(self, state, action, reward):
        max_next_Q = np.max(self.q_table[action, :])
        self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * (reward + self.discount_factor * max_next_Q - self.q_table[state, action])
        
        # TODO: Extract the logging from this function
        plt.imshow(self.q_table, cmap='hot', interpolation='nearest')
        plt.title('Q Table')

        # Save the plot to a file
        heatmap_file = 'q_table.png'
        plt.savefig(heatmap_file)

        # Log the heatmap image to wandb
        wandb.log({"q_table": wandb.Image(heatmap_file)})

    def get_optimal(self):
        best_state_idx = np.unravel_index(np.argmax(self.q_table), self.q_table.shape)[0]
        best_hyperparameters = state_space[best_state_idx]
        return best_hyperparameters

optimiser_model = RL(state_space)

# Set up the interpretability model
# TODO: Get these sizes from the task 
interpretability_model = Transformer(11403, (2, 2)).to(DEVICE)

# TODO: Use an evaluator structure instead of function on task

# Train the RL model

# TODO: Commonise this with train_subject_model_batch
def train_subject_models(task, model, trainer, count):
    nets = [model(task).to(DEVICE) for _ in range(count)]
    targets = [task.get_dataset(i).get_target() for i in range(count)]
    losses = trainer.train_parallel(
        nets,
        [task.get_dataset(i) for i in range(count)],
        [task.get_dataset(i, type=VAL) for i in range(count)],
    )

    for i, (net, target, loss) in enumerate(zip(nets, targets, losses)):
        net_id = uuid.uuid4()
        model_path = f'{SUBJECT_MODEL_PATH}/{net_id}.pickle'
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
        with open(f'{SUBJECT_MODEL_PATH}/index.txt', 'a') as md_file:
            md_file.write(json.dumps(md) + '\n')
    
    subject_model_loss = sum(losses) / len(losses)
    return subject_model_loss


def train_interpretability_model(interpretability_model, subject_model_names, train_ratio=0.7):
    train_sample_count = int(len(subject_model_names) * train_ratio)
    wandb.log({'subject_model_count': train_sample_count})
    train_dataset = MultifunctionSubjectModelDataset(SUBJECT_MODEL_PATH, subject_model_names[:train_sample_count])
    train_dataloader = DataLoader(train_dataset, batch_size=INTERPRETABILITY_BATCH_SIZE, shuffle=True, num_workers=0)
    test_dataset = MultifunctionSubjectModelDataset(SUBJECT_MODEL_PATH, subject_model_names[train_sample_count:])
    test_dataloader = DataLoader(test_dataset, batch_size=INTERPRETABILITY_BATCH_SIZE, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(interpretability_model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()
    model = Transformer(train_dataset.model_param_count, train_dataset.output_shape).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    interpretability_model.train()
    for _ in range(INTERPRETABILITY_MODEL_EPOCHS):
        interpretability_model.eval()
        test_loss = 0.0
        with torch.no_grad():
            eval_dict = {'op1': [], 'op2': [], 'predicted_op1': [], 'predicted_op2': []}
            for inputs, targets in test_dataloader:
                outputs = interpretability_model(inputs.to(DEVICE))
                loss = criterion(outputs, targets.to(DEVICE))
                test_loss += loss.item() * inputs.size(0)
                for output, target in zip(outputs, targets):
                    d = task.decode(target)
                    eval_dict['op1'].append(d[0][1])
                    eval_dict['op2'].append(d[1][1])
                    predicted_d = task.decode(output)
                    eval_dict['predicted_op1'].append(predicted_d[0][1])
                    eval_dict['predicted_op2'].append(predicted_d[1][1])
            data = list(zip(*eval_dict.values()))
            columns = list(eval_dict.keys())
            interpretability_model_predictions = wandb.Table(data=data, columns=columns)
            wandb.log({"interpretability_model_predictions": interpretability_model_predictions})
        avg_loss = test_loss / len(test_dataset)

        for (inputs, targets) in train_dataloader:
            optimizer.zero_grad()
            outputs = interpretability_model(inputs.to(DEVICE))
            loss = criterion(outputs, targets.to(DEVICE))
            loss.backward()
            optimizer.step()

    return avg_loss


os.environ["WAND_API_KEY"] = "dd685d7aa9b38a2289e5784a961b81e22fc4c735"
wandb.init(project='bounding-mi', entity='patrickaaleask', reinit=True)

for episode in range(EPISODES):
    print(f'Episode {episode} of {EPISODES}')
    state = np.random.randint(len(state_space))

    for step in range(STEPS):
        print(f'Step {step} of {STEPS}')
        action = optimiser_model.get_action(state)
        hp = state_space[action]

        trainer = AdamTrainer(task, SUBJECT_MODEL_EPOCHS, SUBJECT_MODEL_BATCH_SIZE, weight_decay=hp[0], device=DEVICE, lr=hp[1])
        subject_model_loss = train_subject_models(task, IntegerGroupFunctionRecoveryModel, trainer, SUBJECT_MODELS_PER_STEP)

        subject_model_names = get_matching_subject_models_names(SUBJECT_MODEL_PATH, task=task)
        interpretability_model_loss = train_interpretability_model(interpretability_model, subject_model_names)

        reward = -(INTERPRETABILITY_WEIGHT * interpretability_model_loss + (1 - INTERPRETABILITY_WEIGHT) * subject_model_loss)
        optimiser_model.update(state, action, reward)
        
        state = action

        wandb.log({
            'interpretability_loss': interpretability_model_loss,
            'subject_model_loss': subject_model_loss,
        })

# Return the best hyperparameters
print(optimiser_model.get_optimal())
print(f"Optimal weight decay is {optimiser_model.get_optimal()[0]}, optimal lr is {optimiser_model.get_optimal()[1]}")