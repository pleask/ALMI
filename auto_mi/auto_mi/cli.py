"""
CLI tools for running automated interpretability experiments.
"""
import argparse
import random

import wandb

from auto_mi.trainers import AdamTrainer
from auto_mi.mi import Transformer, train_mi_model
from auto_mi.rl import pretrain_subject_models
from auto_mi.utils import evaluate_subject_model


def train_cli(
    subject_model_io_class,
    interpretability_model_io_class,
    task_class,
    subject_model_class,
    default_subject_model_epochs=100,
    default_subject_model_lr=0.01,
    default_subject_model_batch_size=1000,
    default_interpretability_model_num_layers=2,
    default_interpretability_model_num_heads=16,
    default_interpretability_model_positional_encoding_size=1024,
    default_interpretability_model_batch_size=2**5,
    default_interpretability_model_subject_model_count=-1,
):
    """
    CLI wrapper for training subject models and interpretability models for a task set up.
    """
    parser = argparse.ArgumentParser(
        description="Run either pretraining or the full pipeline."
    )

    global_group = parser.add_argument_group("Global Arguments")
    global_group.add_argument("--seed", type=float, help="Random seed.", default=0.0)
    global_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train or evaluate models on",
    )

    subject_model_group = parser.add_argument_group("Subject Model Arguments")
    subject_model_group.add_argument(
        "--evaluate_subject_models",
        action="store_true",
        help="Evaluate a subject model.",
    )
    subject_model_group.add_argument(
        "--train_subject_models", action="store_true", help="Train the subject models."
    )
    subject_model_group.add_argument(
        "--subject_model_count",
        type=int,
        help="Number of subject models to train.",
        default=10,
    )
    subject_model_group.add_argument(
        "--subject_model_path", type=str, help="Path of the subject models"
    )
    subject_model_group.add_argument(
        "--subject_model_epochs",
        type=str,
        help="Number of epochs to train the subject models.",
        default=default_subject_model_epochs,
    )
    subject_model_group.add_argument(
        "--subject_model_lr",
        type=str,
        help="Learning rate for training the subject models",
        default=default_subject_model_lr,
    )
    subject_model_group.add_argument(
        "--subject_model_batch_size",
        type=str,
        help="Learning rate for training the subject models",
        default=default_subject_model_batch_size,
    )

    interpretability_model_group = parser.add_argument_group(
        "Interpretability Model Arguments"
    )
    interpretability_model_group.add_argument(
        "--interpretability_model_path",
        type=str,
        help="Path of the interpretability models",
    )
    interpretability_model_group.add_argument(
        "--interpretability_model_num_layers",
        type=str,
        help="Number of layers for the transformer model.",
        default=default_interpretability_model_num_layers,
    )
    interpretability_model_group.add_argument(
        "--interpretability_model_num_heads",
        type=str,
        help="Number of layers for the transformer model.",
        default=default_interpretability_model_num_heads,
    )
    interpretability_model_group.add_argument(
        "--interpretability_model_positional_encoding_size",
        type=str,
        help="Number of layers for the transformer model.",
        default=default_interpretability_model_positional_encoding_size,
    )
    interpretability_model_group.add_argument(
        "--interpretability_model_batch_size",
        type=str,
        help="Number of layers for the transformer model.",
        default=default_interpretability_model_batch_size,
    )
    interpretability_model_group.add_argument(
        "--interpretability_model_subject_model_count",
        type=str,
        help="Number of layers for the transformer model.",
        default=default_interpretability_model_subject_model_count,
    )
    args = parser.parse_args()

    subject_model_io = subject_model_io_class(args.subject_model_path)
    interpretability_model_io = interpretability_model_io_class(
        args.interpretability_model_path
    )

    if args.evaluate_subject_models:
        evaluate_subject_model(task_class, subject_model_class, subject_model_io)
        quit()

    task = task_class(args.seed)
    trainer = AdamTrainer(
        task,
        args.subject_model_epochs,
        args.subject_model_batch_size,
        lr=args.subject_model_lr,
        device=args.device,
    )

    sample_model = subject_model_class(task)
    subject_model_parameter_count = sum(p.numel() for p in sample_model.parameters())
    print("Layer parameters")
    print(f"Subject model parameter count: {subject_model_parameter_count}", flush=True)

    if args.train_subject_models:
        state_space = [trainer]

        print("Pretraining subject models")
        trainer = random.choice(state_space)
        pretrain_subject_models(
            trainer,
            subject_model_io,
            subject_model_class,
            task,
            batch_size=args.subject_model_batch_size,
        )
    else:
        wandb.init(
            project="bounding-mi",
            entity="patrickaaleask",
            reinit=True,
            tags=["diag_straight"],
        )

        interpretability_model = Transformer(
            subject_model_parameter_count,
            task.mi_output_shape,
            num_layers=args.interpretability_model_num_layers,
            num_heads=args.interpretability_model_num_heads,
            positional_encoding_size=args.interpretability_model_positional_encoding_size,
        ).to(args.device)
        interpretability_model_parameter_count = sum(
            p.numel() for p in interpretability_model.parameters()
        )
        print(
            f"Interpretability model parameter count: {interpretability_model_parameter_count}"
        )
        train_mi_model(
            interpretability_model,
            interpretability_model_io,
            subject_model_class,
            subject_model_io,
            trainer,
            task,
            device=args.device,
            batch_size=args.interpretability_model_batch_size,
            subject_model_count=args.interpretability_model_subject_model_count,
        )
