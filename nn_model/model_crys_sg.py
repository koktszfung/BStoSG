import torch.nn as nn


def get_base_model(*args):
    layers = []
    for i in range(len(args) - 1):
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(args[i], args[i+1]))
    model = nn.Sequential(*layers)
    return model
