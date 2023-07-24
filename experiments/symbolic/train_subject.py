import torch
import random

from auto_mi.models import SimpleFunctionRecoveryModel
from auto_mi.tasks import SymbolicFunctionRecoveryTask
from auto_mi.trainers import AdamTrainer
from auto_mi.utils import train_subject_model_batch, get_args_for_slum


# cpu actually seems to be faster for these small models (perhaps as less frequent transfers to gpu)
DEVICE = torch.device("cpu")


if __name__ == "__main__":
    args = get_args_for_slum()
    random.seed(a=args.seed)

    epochs = 10
    batch_size = 1024
    weight_decay = random.choice([0., 0.0001, 0.001, 0.01, 0.1, 1])
    prune_amount = random.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    task = SymbolicFunctionRecoveryTask(args.seed)
    model = SimpleFunctionRecoveryModel
    trainer = AdamTrainer(task, epochs, batch_size, weight_decay=weight_decay, prune_amount=prune_amount, device=DEVICE)

    train_subject_model_batch(task, model, trainer, args.seed, args.start_idx, args.end_idx, args.path)