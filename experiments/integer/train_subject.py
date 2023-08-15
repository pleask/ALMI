import torch
import random

from auto_mi.models import IntegerGroupFunctionRecoveryModel
from auto_mi.tasks import IntegerGroupFunctionRecoveryTask
from auto_mi.trainers import AdamTrainer
from auto_mi.utils import train_subject_model_batch, get_args_for_slum


# cpu actually seems to be faster for these small models (perhaps as less frequent transfers to gpu)
DEVICE = torch.device("cpu")


if __name__ == "__main__":
    args = get_args_for_slum()
    random.seed(a=args.seed)

    epochs = 300
    batch_size = 2**11
    # weight_decay = random.choice([0., 0.0001, 0.001, 0.01, 0.1, 1])
    # prune_amount = random.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    weight_decay = 0.0001
    prune_amount = 0.
    lr = 0.01

    task = IntegerGroupFunctionRecoveryTask(2**3 - 1, 6, seed=args.seed)
    model = IntegerGroupFunctionRecoveryModel
    trainer = AdamTrainer(task, epochs, batch_size, weight_decay=weight_decay, prune_amount=prune_amount, device=DEVICE, lr=lr)

    train_subject_model_batch(task, model, trainer, args.seed, args.start_idx, args.end_idx, args.path, device=DEVICE)
