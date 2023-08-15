import torch
import random

from auto_mi.models import ConvMNIST
from auto_mi.tasks import TrojanMNISTTask
from auto_mi.trainers import AdamTrainer
from auto_mi.utils import train_subject_model_batch, get_args_for_slum


DEVICE = torch.device("cpu")


if __name__ == "__main__":
    args = get_args_for_slum()
    random.seed(a=args.seed)

    epochs = 100
    batch_size = 1024
    lr = 0.001
    weight_decay = random.choice([0., 0.0001, 0.001, 0.01, 0.1, 1])
    prune_amount = random.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    prune_amount = 0.
    weight_decay = 0.001

    task = TrojanMNISTTask(args.seed)
    # model has ~10k parameters like this
    model = ConvMNIST
    trainer = AdamTrainer(task, epochs, batch_size, weight_decay=weight_decay, prune_amount=prune_amount, device=DEVICE, lr=lr)

    train_subject_model_batch(task, model, trainer, args.seed, args.start_idx, args.end_idx, args.path, device=DEVICE)