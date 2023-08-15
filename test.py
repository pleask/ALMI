from auto_mi.tasks import TrojanMNISTTask, VAL, IntegerGroupFunctionRecoveryTask
from auto_mi.models import IntegerGroupFunctionRecoveryModel, ConvMNIST
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

t = IntegerGroupFunctionRecoveryTask(7, 6)
model = ConvMNIST(t)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))