import torch
import random
import json
import argparse
import uuid
from time import gmtime, strftime

from auto_mi.tasks import VAL, SimpleFunctionRecoveryTask, SymbolicFunctionRecoveryTask
from auto_mi.models import SimpleFunctionRecoveryModel
from auto_mi.trainers import AdamTrainer, AdamL1UnstructuredPruneTrainer, AdamWeightDecayTrainer


# cpu actually seems to be faster for these small models (perhaps as less frequent transfers to gpu)
DEVICE = torch.device("cpu")


parser = argparse.ArgumentParser(
    description="Trains subject models, ie. the models that implement the labeling function."
)
parser.add_argument("--path", type=str, help="Directory to which to save the models")
parser.add_argument(
    "--task",
    type=str,
    default='SimpleFunctionRecoveryTask',
    help="Which task to train the subject models on"
)
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


def assert_is_unique_model(index_file, seed, index, task, model, trainer):
    try:
        with open(index_file, 'r') as idx_file:
            for line in idx_file:
                md = json.loads(line.strip())
                if not(md['seed'] == seed and md['index'] != index):
                    continue
                if md['task'] != task.get_metadata():
                    continue
                if md['model'] != model.get_metadata():
                    continue
                if md['trainer'] != trainer.get_metadata():
                    continue
                raise ValueError(f'Model already exists as {md["id"]}')
    except FileNotFoundError:
        return


if __name__ == "__main__":
    print('Parsing args')
    args = parser.parse_args()
    random.seed(a=args.seed)

    task = SimpleFunctionRecoveryTask(args.seed)
    if args.task == SymbolicFunctionRecoveryTask.__name__:
        task = SymbolicFunctionRecoveryTask(args.seed)
    model = SimpleFunctionRecoveryModel
    
    if args.prune_amount > 0.:
        trainer = AdamL1UnstructuredPruneTrainer(task, args.epochs, 1024, device=DEVICE, prune_amount=args.prune_amount)
    elif args.weight_decay > 0.:
        trainer = AdamWeightDecayTrainer(task, args.epochs, 1024, device=DEVICE, weight_decay=args.weight_decay)
    else:
        trainer = AdamTrainer(task, args.epochs, 1024, device=DEVICE)

    for idx in range(args.start_index, args.end_index):
        print(f'Training model {idx} of {args.start_index} to {args.end_index}')
        net = model(task).to(DEVICE)
        trainer.train(net, task.get_dataset(idx))
        loss = trainer.evaluate(net, task.get_dataset(idx, type=VAL))

        assert_is_unique_model(f'{args.path}/index.txt', args.seed, idx, task, net, trainer)

        net_id = uuid.uuid4()
        md = vars(args)
        md['task'] = task.get_dataset(idx).get_metadata()
        md['model'] = net.get_metadata()
        md['trainer'] = trainer.get_metadata()
        md["loss"] = loss
        md["id"] = str(net_id)
        md["time"] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        md["index"] = idx

        with open(f'{args.path}/index.txt', 'a') as md_file:
            md_file.write(json.dumps(md) + '\n')

        model_path = get_model_path(args.path, net_id) 
        torch.save(net.state_dict(), model_path)
