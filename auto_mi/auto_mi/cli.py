"""
CLI tools for running automated interpretability experiments.
"""
import argparse
import wandb

from auto_mi.trainers import AdamTrainer
from auto_mi.mi import train_mi_model, pretrain_mi_model
from auto_mi.subject_models import evaluate_subject_model, train_subject_models


def train_cli(
    tags,
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
        help="Number of subject models to train or evaluate.",
        default=10,
    )
    subject_model_group.add_argument(
        "--subject_model_path", type=str, help="Path of the subject models"
    )
    subject_model_group.add_argument(
        "--subject_model_epochs",
        type=int,
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
        help="Batch size for training the subject models",
        default=default_subject_model_batch_size,
    )
    subject_model_group.add_argument(
        "--subject_model_num_classes",
        type=int,
        help="Number of classes to use from the training data.",
        required=True,
    )
    subject_model_group.add_argument(
        "--subject_model_example_count",
        type=int,
        help="Number of examples to use from the training data for training the subject models. If -1, use all examples.",
        default=-1,
    )
    subject_model_group.add_argument(
        "--subject_model_variant",
        type=int,
        help="Which variant of the subject model to use. Between 0 and 99 inclusive.",
        default=0,
    )
    subject_model_group.add_argument(
        "--example_start_index",
        type=int,
        help="Example index from which to start.",
        default=0,
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
        type=int,
        help="Number of layers for the transformer model.",
        default=default_interpretability_model_num_layers,
    )
    interpretability_model_group.add_argument(
        "--interpretability_model_num_heads",
        type=int,
        help="Number of layers for the transformer model.",
        default=default_interpretability_model_num_heads,
    )
    interpretability_model_group.add_argument(
        "--interpretability_model_positional_encoding_size",
        type=int,
        help="Number of layers for the transformer model.",
        default=default_interpretability_model_positional_encoding_size,
    )
    interpretability_model_group.add_argument(
        "--interpretability_model_batch_size",
        type=int,
        help="Number of layers for the transformer model.",
        default=default_interpretability_model_batch_size,
    )
    interpretability_model_group.add_argument(
        "--interpretability_model_subject_model_count",
        type=int,
        help="Number of subject models to use for training the interpretability model.",
        default=default_interpretability_model_subject_model_count,
    )
    interpretability_model_group.add_argument(
        "--evaluate_interpretability_model",
        type=str,
        help="Evaluate an interpretability model.",
    )
    interpretability_model_group.add_argument(
        "--interpretability_model_lr",
        type=float,
        default=1e-5,
        help="Learning rate for training the interpretability model.",
    )
    interpretability_model_group.add_argument(
        "--interpretability_model_resume",
        type=str,
        default="",
        help="W&B run ID of a run to resume training.",
    )
    subject_model_group.add_argument(
        "--interpretability_model_epochs",
        type=int,
        help="Number of epochs to train the interpretability models.",
        default=default_subject_model_epochs,
    )
    interpretability_model_group.add_argument(
        "--interpretability_model_split_on_variants",
        action="store_true",
        help="Whether to split on variants, i.e. evaluate the interpretability model on a disjoint set of subject model variants to those used for training.",
    )
    interpretability_model_group.add_argument(
        "--interpretability_model_variant_range_start",
        type=int,
        default=-1,
        help="If specified, the start of the variant range to use.",
    )
    interpretability_model_group.add_argument(
        "--interpretability_model_variant_range_end",
        type=int,
        default=-1,
        help="If specified, the end of the variant range to use.",
    )
    interpretability_model_group.add_argument(
        "--interpretability_model_pretrain",
        action="store_true",
        help="Pretrain the interpretability model on the next token prediction task.",
    )

    args = parser.parse_args()

    subject_model_io = subject_model_io_class(args.subject_model_path)
    interpretability_model_io = interpretability_model_io_class(
        args.interpretability_model_path
    )
    task = task_class(
        seed=args.seed,
        num_classes=args.subject_model_num_classes,
        num_examples=args.subject_model_example_count,
    )
    sample_model = subject_model_class(task, variant=args.subject_model_variant)
    subject_model_parameter_count = sum(p.numel() for p in sample_model.parameters())
    print(f"Subject model parameter count: {subject_model_parameter_count}", flush=True)

    trainer = AdamTrainer(
        task,
        args.subject_model_epochs,
        args.subject_model_batch_size,
        lr=args.subject_model_lr,
        device=args.device,
    )

    if args.evaluate_subject_models:
        evaluate_subject_model(task, subject_model_class, subject_model_io, trainer, model_count=args.subject_model_count)
        quit()

    if args.train_subject_models:
        print("Pretraining subject models")
        train_subject_models(
            task,
            subject_model_class,
            trainer,
            subject_model_io,
            count=args.subject_model_count,
            variant=args.subject_model_variant,
            start_example=args.example_start_index,
        )
        quit()

    wandb_kwargs = {}
    if args.interpretability_model_resume:
        wandb_kwargs["resume"] = True
        wandb_kwargs["id"] = args.interpretability_model_resume
    run = wandb.init(
        project="bounding-mi",
        entity="patrickaaleask",
        reinit=True,
        tags=tags,
        **wandb_kwargs,
    )
    wandb.config.update(args, allow_val_change=True)
    wandb.config.update({"num_classes": task.output_shape[0]})
    wandb.save(subject_model_io.index_file)

    if args.interpretability_model_pretrain:
        pretrain_mi_model(
            run,
            interpretability_model_io,
            subject_model_class,
            subject_model_io,
            trainer,
            task,
            device=args.device,
            batch_size=args.interpretability_model_batch_size,
            subject_model_count=args.interpretability_model_subject_model_count,
            lr=args.interpretability_model_lr,
            epochs=args.interpretability_model_epochs,
            split_on_variants=args.interpretability_model_split_on_variants,
            variant_range_start=args.interpretability_model_variant_range_start,
            variant_range_end=args.interpretability_model_variant_range_end,
            num_layers=args.interpretability_model_num_layers,
            num_heads=args.interpretability_model_num_heads,
            positional_encoding_size=args.interpretability_model_positional_encoding_size,
            load_interpretability_model=args.interpretability_model_resume,
        )
    else:
        train_mi_model(
            run,
            interpretability_model_io,
            subject_model_class,
            subject_model_io,
            trainer,
            task,
            device=args.device,
            batch_size=args.interpretability_model_batch_size,
            subject_model_count=args.interpretability_model_subject_model_count,
            lr=args.interpretability_model_lr,
            epochs=args.interpretability_model_epochs,
            split_on_variants=args.interpretability_model_split_on_variants,
            variant_range_start=args.interpretability_model_variant_range_start,
            variant_range_end=args.interpretability_model_variant_range_end,
            num_layers=args.interpretability_model_num_layers,
            num_heads=args.interpretability_model_num_heads,
            positional_encoding_size=args.interpretability_model_positional_encoding_size,
            load_interpretability_model=args.interpretability_model_resume,
        )
