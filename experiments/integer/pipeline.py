"""
Implements the full meta-learning pipeline.
"""
import math

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from auto_mi.tasks import IntegerGroupFunctionRecoveryTask, VAL
from auto_mi.base import MetadataBase
from auto_mi.trainers import AdamTrainer
from auto_mi.mi import Transformer

EPISODES = 20
STEPS = 5
SUBJECT_MODEL_EPOCHS = 10
SUBJECT_MODEL_BATCH_SIZE = 2**10
SUBJECT_MODELS_PER_STEP = 10
INTERPRETABILITY_WEIGHT = 0.5
DEVICE = 'cuda'
INTERPRETABILITY_BATCH_SIZE = 128
INTERPRETABILITY_MODEL_EPOCHS = 20

class IntegerGroupFunctionRecoveryModel(nn.Module, MetadataBase):
    def __init__(self, task):
        super().__init__()

        flattened_input_size = math.prod(task.input_shape)
        hidden_layer_size = 100
        self.l1 = nn.Linear(flattened_input_size, hidden_layer_size)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.r2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_layer_size, task.output_shape[0])

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.l1(x)
        x = self.r1(x)
        x = self.l2(x)
        x = self.r2(x)
        x = self.l3(x)
        return torch.sigmoid(x)

# Define the subject model task we're trying to solve here
task = IntegerGroupFunctionRecoveryTask(2**3 - 1, 3)

# Define the hyperparameter search space
hyperparameters = {
    'weight_decay': [0, 0.1, 0.01, 0.001, 0.0001],
    'learning_rate': [0.1, 0.01, 0.001, 0.0001],
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
        print(self.q_table)

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

class SubjectModelDataset(Dataset):
    def __init__(self, nets, targets):
        super().__init__()
        self.nets = nets
        self.targets = targets
        self.device = DEVICE

    def __len__(self):
        return len(self.nets)

    def __getitem__(self, idx):
        net = self.nets[idx]
        x = torch.concat(
            [param.detach().reshape(-1) for _, param in net.named_parameters()]
        ).to(self.device)

        y = self.targets[idx]
        return x, y

    @property
    def model_param_count(self):
        return self[0][0].shape[0]

    @property
    def output_shape(self):
        return self[0][1].shape


def train_subject_models(task, model, trainer, count):
    nets = []
    targets = []
    losses = []
    for i in range(count):
        print(f'Training subject model {i} of {count}')
        net = model(task).to(DEVICE)
        loss = trainer.train(net, task.get_dataset(i), task.get_dataset(i, type=VAL))

        nets.append(net)
        targets.append(task.get_dataset(i).get_target())
        losses.append(loss)
    
    subject_model_loss = sum(losses) / len(losses)
    return nets, targets, subject_model_loss


# XXX: Does this lead to catastrophic forgetting?
def train_interpretability_model(interpretability_model, subject_models, subject_targets, train_ratio=0.7):
    train_sample_count = int(len(subject_models) * train_ratio)
    train_dataset = SubjectModelDataset(subject_models[:train_sample_count], subject_targets[:train_sample_count])
    train_dataloader = DataLoader(train_dataset, batch_size=INTERPRETABILITY_BATCH_SIZE, shuffle=True, num_workers=0)
    test_dataset = SubjectModelDataset(subject_models[train_sample_count:], subject_targets[train_sample_count:])
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
            for inputs, targets in test_dataloader:
                outputs = interpretability_model(inputs.to(DEVICE))
                loss = criterion(outputs, targets.to(DEVICE))
                test_loss += loss.item() * inputs.size(0)
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
    print(f'Episode {episode}')
    state = np.random.randint(len(state_space))

    for step in range(STEPS):
        action = optimiser_model.get_action(state)
        hp = state_space[action]

        trainer = AdamTrainer(task, SUBJECT_MODEL_EPOCHS, SUBJECT_MODEL_BATCH_SIZE, weight_decay=hp[0], device=DEVICE, lr=hp[1])
        subject_models, subject_targets, subject_model_loss = train_subject_models(task, IntegerGroupFunctionRecoveryModel, trainer, SUBJECT_MODELS_PER_STEP)

        interpretability_model_loss = train_interpretability_model(interpretability_model, subject_models, subject_targets)

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