import torch
import random
import json
import argparse
import uuid
from time import gmtime, strftime

from auto_mi.tasks import VAL, SimpleFunctionRecoveryTask
from auto_mi.models import SimpleFunctionRecoveryModel
from auto_mi.trainers import AdamTrainer, AdamL1UnstructuredPruneTrainer, AdamWeightDecayTrainer


# cpu actually seems to be faster for these small models (perhaps as less frequent transfers to gpu)
DEVICE = torch.device("cpu")


parser = argparse.ArgumentParser(
    description="Trains subject models, ie. the models that implement the labeling function."
)
parser.add_argument("--path", type=str, help="Directory to which to save the models")
parser.add_argument(
    "--seed",
    type=int,
    help="The random seed to use. Should be different for all runs of this script within an experiment, and the same for each run across experiments.",
)
parser.add_argument(
    "--start_index",
    type=int,
    help="The start index of the subject models.",
    required=True
)
parser.add_argument(
    "--end_index",
    type=int,
    help="The end index of the subject models (non-inclusive).",
    required=True
)
parser.add_argument(
    "--epochs", type=int, help="The number of epochs for which to train the models."
)
parser.add_argument(
    "--weight_decay", default=0., type=float, help="Weight decay for the adam optimiser."
)
parser.add_argument(
    "--prune_amount", default=0., type=float, help="Amount to L1 unstructured prune the models."
)


get_model_path = lambda path, net_idx: f"{path}/{net_idx}.pickle"


def assert_is_unique_model(path, seed, index, task, model, epochs, weight_decay, prune_amount):
    try:
        with open(f'{path}/index.txt', 'r') as idx_file:
            for line in idx_file:
                md = json.loads(line.strip())
                if md['seed'] == seed and md['index'] == index and md['task'] == task and md['model'] == model and md['epochs'] == epochs and md['weight_decay'] == weight_decay and md['prune_amount'] == prune_amount:
                    raise ValueError(f'Model already exists as {md["id"]}')
    except FileNotFoundError:
        return


if __name__ == "__main__":
    print('Parsing args')
    args = parser.parse_args()
    random.seed(a=args.seed)

    task = SimpleFunctionRecoveryTask(args.seed)
    model = SimpleFunctionRecoveryModel
    
    if args.prune_amount > 0.:
        trainer = AdamL1UnstructuredPruneTrainer(task, args.epochs, 1024, device=DEVICE, prune_amount=args.prune_amount)
    elif args.weight_decay > 0.:
        trainer = AdamWeightDecayTrainer(task, args.epochs, 1024, device=DEVICE, weight_decay=args.weight_decay)
    else:
        trainer = AdamTrainer(task, args.epochs, 1024, device=DEVICE)

    for idx in range(args.start_index, args.end_index):
        assert_is_unique_model(args.path, args.seed, idx, type(task).__name__, type(model).__name__, args.epochs, args.weight_decay, args.prune_amount)

        print(f'Training model {idx} of {args.start_index} to {args.end_index}')
        net = model(task).to(DEVICE)
        trainer.train(net, task.get_dataset(idx))
        loss = trainer.evaluate(net, task.get_dataset(idx, type=VAL))

        net_id = uuid.uuid4()
        md = vars(args)
        md.update(task.get_dataset(idx).get_metadata())
        md["loss"] = loss
        md["id"] = str(net_id)
        md["time"] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        md["index"] = idx

        with open(f'{args.path}/index.txt', 'a') as md_file:
            md_file.write(json.dumps(md) + '\n')

        model_path = get_model_path(args.path, net_id) 
        torch.save(net.state_dict(), model_path)
