import torch.nn as nn

class SimpleFunctionRecoveryModel(nn.Module):
    def __init__(self, task):
        super().__init__()

        hidden_layer_size = 25
        self.l1 = nn.Linear(task.input_shape[0], hidden_layer_size)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.r2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_layer_size, task.output_shape[0])

    def forward(self, x):
        return self.l3(self.r2(self.l2(self.r1(self.l1(x)))))
        