import torch.nn as nn

def get_auto_model(task, hidden_layer_size=25):
    """
    Automatically gets an appropriate(ish) model for the task.
    """
    return nn.Sequential(
        nn.Linear(task.input_shape, hidden_layer_size),
        nn.ReLU(),
        nn.Linear(hidden_layer_size, hidden_layer_size),
        nn.ReLU(),
        nn.Linear(hidden_layer_size, task.output_shape),
    )