from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import wandb

from auto_mi.utils import train_subject_models
from auto_mi.mi import train_interpretability_model


class BaseQLearner(ABC):
    @abstractmethod
    def __init__(self, state_space, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        pass

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def update(self, state, action, reward):
        pass

    @abstractmethod
    def get_optimal(self):
        pass


class QLearner:
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
        best_state = self.state_space[best_state_idx]
        return best_state


def train_optimiser_model(optimiser_model, interpretability_model, subject_model_path, subject_model, task, episodes, steps, subject_models_per_step=10, interpretability_weight=0.5):
    for episode in range(episodes):
        print(f'Episode {episode} of {episodes}')
        state = np.random.randint(len(optimiser_model.state_space))
        for step in range(steps):
            print(f'Step {step} of {steps}')
            action = optimiser_model.get_action(state)
            trainer = optimiser_model.state_space[action]

            # Use the current trainer to train new subject models
            subject_model_loss = train_subject_models(task, subject_model, trainer, subject_model_path, count=subject_models_per_step)

            # Train the interpretability model using the new subject models and existing subject models
            interpretability_model_loss, validation_loss = train_interpretability_model(interpretability_model, task, subject_model_path)

            reward = -(interpretability_weight * interpretability_model_loss + (1 - interpretability_weight) * subject_model_loss)

            state = action

            wandb.log({
                'interpretability_loss': interpretability_model_loss,
                'subject_model_loss': subject_model_loss,
                'reward': reward,
                'validation_interpretability_loss': validation_loss,
            })