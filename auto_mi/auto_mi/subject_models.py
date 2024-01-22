from collections import Counter
import random
from time import gmtime, strftime
import uuid

import torch

from auto_mi.tasks import VAL

TRAIN_RATIO = .7


def train_subject_models(task, model, trainer, model_writer, count=10, device="cpu"):
    """
    Trains subject models using the specified trainer. Returns the average loss
    of the subject models, and a sub-group of the trained subject models that
    are used to validate the performance of the interpretability model on
    subject models created by this trainer.
    """
    nets = [model(task).to(device) for _ in range(count)]
    targets = [task.get_dataset(i).get_target() for i in range(count)]
    losses, train_losses = trainer.train_parallel(
        nets,
        [task.get_dataset(i) for i in range(count)],
        [task.get_dataset(i, type=VAL) for i in range(count)],
    )

    print('Writing models to disk...')
    model_ids = []
    for i, (net, target, loss, train_loss) in enumerate(
        zip(nets, targets, losses, train_losses)
    ):
        net_id = str(uuid.uuid4())
        model_writer.write_metadata(
            md={
                "task": task.get_metadata(),
                "example": task.get_dataset(i).get_metadata(),
                "model": net.get_metadata(),
                "trainer": trainer.get_metadata(),
                "loss": loss,
                "train_loss": train_loss,
                "id": str(net_id),
                "time": strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                "index": i,
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
    subject_model_names, _ = get_matching_subject_models_names(subject_model_io, trainer, task)
    if len(subject_model_names) > model_count:
        subject_model_names = random.sample(subject_model_names, model_count)
    metadata = [md for md in metadata if md["id"] in subject_model_names]
    assert len(metadata) != 0

    accuracies = []
    # TODO: Use random models
    for model_idx in range(min(model_count, len(metadata))):
        print(f"Model {model_idx}")
        task.seed = metadata[model_idx]["task"]["seed"]
        example = task.get_dataset(metadata[model_idx]["index"], type=VAL)
        model_id = metadata[model_idx]["id"]
        permutation_map = metadata[model_idx]["example"]["permutation_map"]
        print(f"Permutation map: {permutation_map}")
        model = subject_model_io.get_model(subject_model_class(), model_id)
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
    frozen_layers=None,
    subject_model_example_count=-1,
):
    """
    Returns a list of subject model names that match the specified trainer and
    task by searching the index.txt file.

    frozen_layers: If set to a tuple, returns only models that
    have the layers specified in the tuple frozen.
    """
    matching_subject_models_names = []
    losses = []
    metadata = model_writer.get_metadata()

    reasons = Counter()
    for md in metadata:
        if md["task"]["name"] != type(task).__name__:
            reasons['task.name'] += 1
            continue

        trainer_metadata = trainer.get_metadata()
        if not all(
            md["trainer"][key] == trainer_metadata[key]
            for key in ["name", "weight_decay", "lr", "l1_penalty_weight"]
        ):
            reasons['trainer'] += 1
            continue

        if md["id"] in exclude:
            reasons['exclude'] += 1 
            continue

        # Have had a few issues where model pickles aren't saved but their
        # metadata is still written, so skip those models.
        if not model_writer.check_model_exists(md["id"]):
            reasons['model_exists'] += 1   
            continue

        if frozen_layers is not None:
            try:
                if set(md["model"]["frozen"]) != set(frozen_layers):
                    reasons['frozen_layers'] += 1
                    continue
            except KeyError:
                continue

        try:
            if md["task"]["num_classes"] != task.num_classes:
                reasons['num_classes'] += 1
                continue
        except KeyError:
            # Older datasets will not have num_classes in their metadata
            # TODO: Remove this once all models have num_classes in their metadata
            pass

        try:
            if md["task"]["num_examples"] != task.num_examples:
                reasons['num_examples'] += 1
                continue
        except KeyError:
            # Older datasets will not have num_examples in their metadata
            # TODO: Remove this once all models have num_examples in their metadata
            pass

        matching_subject_models_names.append(md["id"])
        losses.append(md["loss"])

    print(f"Reasons for exclusion: {reasons}")

    random.shuffle(matching_subject_models_names)
    return matching_subject_models_names, sum(losses) / len(losses) if losses else 0