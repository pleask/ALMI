from abc import ABC, abstractmethod
import random

import matplotlib.pyplot as plt
import numpy as np
import wandb

from auto_mi.utils import train_subject_models
from auto_mi.mi import train_interpretability_model, get_matching_subject_models_names


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
        
        self.log_q_table()

    def log_q_table(self):
        # Visualization of the Q-table
        fig, ax = plt.subplots(figsize=(16, 8))  # Adjust figure size here
        cax = ax.imshow(self.q_table, cmap='hot', interpolation='nearest')
        ax.set_title('Q Table')

        # Adding labels to y-axis
        y_labels = [str(self.state_space[i].get_metadata()) for i in range(len(self.state_space))]
        ax.set_yticks(range(len(self.state_space)))
        ax.set_yticklabels(y_labels) # Adjust rotation and fontsize here

        # Optionally: Add a colorbar
        cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
        cbar.set_label('Q Value', rotation=270, labelpad=15)

        # Adjust subplot params to give labels more space
        plt.subplots_adjust(left=0.25, bottom=0.15, right=0.95, top=0.95)

        # Save the plot to a file
        heatmap_file = 'q_table.png'
        plt.savefig(heatmap_file)
        plt.close()  # Clear the plot to avoid overlay of the images

        # Log the heatmap image to wandb
        wandb.log({
            "q_table": wandb.Image(heatmap_file)
        })


    def get_optimal(self):
        best_state_idx = np.unravel_index(np.argmax(self.q_table), self.q_table.shape)[0]
        best_state = self.state_space[best_state_idx]
        return best_state


def train_optimiser_model(optimiser_model, interpretability_models, model_writer, subject_model, task, episodes, steps, subject_models_per_step=10, interpretability_weight=0.5, should_train_subject_models=False):
    """
    should_train_subject_models: If set to True, train a new batch of subject models that are
    first used for validation. Otherwise, just use the first 1k subject models
    for the trainer for validation, and don't use them in training.
    """
    reward_history = [[] for _ in range(len(optimiser_model.state_space))]
    subject_model_loss_history = [[] for _ in range(len(optimiser_model.state_space))]
    interpretability_model_loss_history = [[] for _ in range(len(optimiser_model.state_space))]

    for episode in range(episodes):
        print(f'Episode {episode} of {episodes}')
        state = np.random.randint(len(optimiser_model.state_space))

        for step in range(steps):
            print(f'Step {step} of {steps}')
            action = optimiser_model.get_action(state)
            trainer = optimiser_model.state_space[action]
            interpretability_model = interpretability_models[action]

            if should_train_subject_models:
                # Use the current trainer to train new subject models
                subject_model_loss, validation_subject_models = train_subject_models(task, subject_model, trainer, model_writer, count=subject_models_per_step, device=interpretability_model.device)
            else:
                # Use the first 1000 subject models in the dataset for validation
                validation_subject_models, subject_model_loss = get_matching_subject_models_names(model_writer, trainer=trainer, task=task)
                validation_subject_models = validation_subject_models[:1000]

            # Train the interpretability model using the new subject models and existing subject models
            interpretability_loss = train_interpretability_model(interpretability_model, task, model_writer, validation_subject_models, trainer)

            reward = -(interpretability_weight * interpretability_loss + (1 - interpretability_weight) * subject_model_loss)
            optimiser_model.update(state, action, reward)

            state = action

            reward_history[state].append(reward)
            subject_model_loss_history[state].append(subject_model_loss)
            interpretability_model_loss_history[state].append(interpretability_loss)

            wandb.log({
                "reward" : wandb.plot.line_series(
                    xs=list(range(max([len(rw) for rw in reward_history]))), 
                    ys=reward_history,
                    keys=[str(trainer.get_metadata()) for trainer in optimiser_model.state_space],
                    title="Reward",
                    xname='Step',
                ),
                "subject_model_loss" : wandb.plot.line_series(
                    xs=list(range(max([len(rw) for rw in subject_model_loss_history]))), 
                    ys=subject_model_loss_history,
                    keys=[str(trainer.get_metadata()) for trainer in optimiser_model.state_space],
                    title="Subject Model Loss",
                    xname='Step',
                ),
                "interpretability_model_loss" : wandb.plot.line_series(
                    xs=list(range(max([len(rw) for rw in interpretability_model_loss_history]))), 
                    ys=interpretability_model_loss_history,
                    keys=[str(trainer.get_metadata()) for trainer in optimiser_model.state_space],
                    title="Interpretability Model Loss",
                    xname='Step',
                ),
            })


def pretrain_subject_models(optimiser_model, model_writer, subject_model, task, batch_size=10):
    """
    Trains random samples of subject models. This is so the dataset generation
    can happen in a highly distributed manner (ie. ~2k CPUs) on Hamilton, rather
    than as part of the pipeline process, which can only access ~64 CPUs.
    """
    trainer = random.choice(optimiser_model.state_space)
    train_subject_models(task, subject_model, trainer, model_writer, count=batch_size)