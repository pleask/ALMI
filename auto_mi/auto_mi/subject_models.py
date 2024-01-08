from auto_mi.mi import TRAIN_RATIO
from auto_mi.tasks import VAL


import uuid
from time import gmtime, strftime


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
    task_class, subject_model_class, subject_model_io, samples=100, model_count=100
):
    metadata = subject_model_io.get_metadata()
    accuracies = []
    # TODO: Use random models
    for model_idx in range(min(model_count, len(metadata))):
        print(f"Model {model_idx}")
        task = task_class(seed=metadata[model_idx]["task"]["seed"])
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