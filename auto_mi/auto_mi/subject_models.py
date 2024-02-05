from collections import Counter
import os
import random
from time import gmtime, strftime
import uuid

import torch

from auto_mi.tasks import VAL

TRAIN_RATIO = 0.7


def train_subject_models(
    task,
    model,
    trainer,
    model_writer,
    count=10,
    device="cpu",
    variant=0,
    start_example=0,
):
    """
    Trains subject models using the specified trainer. Returns the average loss
    of the subject models, and a sub-group of the trained subject models that
    are used to validate the performance of the interpretability model on
    subject models created by this trainer.
    """

    get_idx = lambda i: start_example + i

    nets = [model(task, variant=variant).to(device) for _ in range(count)]

    num_params = sum(p.numel() for p in nets[0].parameters())
    print(f"Subject model param count: {num_params}")

    train_examples = [task.get_dataset(get_idx(i)) for i in range(count)]
    validation_examples = [
        task.get_dataset(start_example + i, type=VAL) for i in range(count)
    ]

    losses, train_losses = trainer.train_parallel(
        nets,
        train_examples,
        validation_examples,
    )

    print("Writing models to disk...")
    model_ids = []
    for i, (net, train_dataset, loss, train_loss) in enumerate(
        zip(nets, train_examples, losses, train_losses)
    ):
        net_id = str(uuid.uuid4())
        model_writer.write_metadata(
            md={
                "task": task.get_metadata(),
                "example": train_dataset.get_metadata(),
                "model": net.get_metadata(),
                "trainer": trainer.get_metadata(),
                "loss": loss,
                "train_loss": train_loss,
                "id": str(net_id),
                "time": strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                "index": get_idx(i),
            }
        )
        model_writer.write_model(net_id, net)

        model_ids.append(str(net_id))

    subject_model_loss = sum(losses) / len(losses)

    validation_subject_model_ids = model_ids[int(TRAIN_RATIO * count) :]

    return subject_model_loss, validation_subject_model_ids


def evaluate_subject_model(
    task, subject_model_class, subject_model_io, trainer, samples=100, model_count=100
):
    metadata = subject_model_io.get_metadata()
    subject_model_names, _ = get_matching_subject_models_names(
        subject_model_io, trainer, task
    )
    if len(subject_model_names) > model_count:
        subject_model_names = random.sample(subject_model_names, model_count)
    metadata = [md for md in metadata if md["id"] in subject_model_names]
    assert len(metadata) != 0

    accuracies = []
    for model_idx in range(min(model_count, len(metadata))):
        print(f"Model {model_idx}")
        example = task.get_dataset(metadata[model_idx]["index"], type=VAL)
        model_id = metadata[model_idx]["id"]
        print(f"Permutation map: {example._permutation_map}")
        try:
            subject_model = subject_model_class(variant=metadata[model_idx]["model"]["variant"])
        except KeyError:
            subject_model = subject_model_class()
        model = subject_model_io.get_model(
            subject_model,
            model_id,
        )
        correct = []
        for _ in range(samples):
            i = random.randint(0, len(example) - 1)
            input, label = example[i]
            prediction = model(torch.Tensor(input).unsqueeze(0))
            correct.append((torch.argmax(prediction, -1) == label)[0].item())
        accuracy = sum(correct) / samples
        print(accuracy)
        accuracies.append(accuracy)
    print("Overall accuracy:", sum(accuracies) / len(accuracies))


def get_matching_subject_models_names(
    model_writer,
    trainer,
    task,
    exclude=[],
    variants=None,
):
    """
    Returns a list of subject model names that match the specified trainer and
    task by searching the index.txt file.
    """
    matching_subject_models_names = []
    losses = []
    metadata = model_writer.get_metadata()

    reasons = Counter()
    for md in metadata:
        if md["task"]["name"] != type(task).__name__:
            reasons["task.name"] += 1
            continue

        trainer_metadata = trainer.get_metadata()
        if not all(
            md["trainer"][key] == trainer_metadata[key]
            for key in ["name", "weight_decay", "lr", "l1_penalty_weight"]
        ):
            reasons["trainer"] += 1
            continue

        if md["id"] in exclude:
            reasons["exclude"] += 1
            continue

        # Have had a few issues where model pickles aren't saved but their
        # metadata is still written, so skip those models.
        if not model_writer.check_model_exists(md["id"]):
            reasons["model_exists"] += 1
            continue

        try:
            if md["task"]["num_classes"] != task.num_classes:
                reasons["num_classes"] += 1
                continue
        except KeyError:
            # Older datasets will not have num_classes in their metadata
            # TODO: Remove this once all models have num_classes in their metadata
            pass

        try:
            if md["task"]["num_examples"] != task.num_examples:
                reasons["num_examples"] += 1
                continue
        except KeyError:
            # Older datasets will not have num_examples in their metadata
            # TODO: Remove this once all models have num_examples in their metadata
            pass

        try:
            if (
                variants is not None
                and variants[0] != -1
                and md["model"]["variant"] not in variants
            ):
                reasons["variant"] += 1
                continue
        except KeyError:
            # Older models will not have variant in their metadata
            pass

        matching_subject_models_names.append(md["id"])
        losses.append(md["loss"])

    print(f"Reasons for exclusion: {reasons}")

    random.shuffle(matching_subject_models_names)
    return matching_subject_models_names, sum(losses) / len(losses) if losses else 0
