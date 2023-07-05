"""
For each subject model, calculates the sparsity of the network, ie. the proportion of parameters that are exactly zero.
"""
import torch
from train_subject_models import get_subject_net

SUBJECT_MODEL_DIR = 'subject_models/'

def get_subject_model(subject_model_name):
    net = get_subject_net()
    net.load_state_dict(torch.load(f"{SUBJECT_MODEL_DIR}/{subject_model_name}.pickle"))
    return net


def get_model_sparsity(model):
    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        if 'weight' in name: 
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()

    sparsity = zero_params / total_params * 100 
    return sparsity

if __name__ == '__main__':
    model_names = []
    with open(f'{SUBJECT_MODEL_DIR}/index.txt', 'r') as idx_file:
        for line in idx_file:
            s = line.strip()
            model_name, _ = line.split(' ', maxsplit=1)
            model_names.append(model_name)

    with open(f'{SUBJECT_MODEL_DIR}/sparsity.txt', 'r') as sparsity_file:
        sparsity_model_names = set()
        for line in sparsity_file:
            s = line.strip()
            sparsity_model_name, _ = line.split(', ', maxsplit=1)
            sparsity_model_names.add(sparsity_model_name)

    for model_name in model_names:
        if model_name in sparsity_model_names:
            continue
        model = get_subject_model(model_name)
        sparsity = get_model_sparsity(model)
        with open(f'{SUBJECT_MODEL_DIR}/sparsity.txt', 'a') as sparsity_file:
            sparsity_file.write(f'{model_name}, {sparsity}\n')
