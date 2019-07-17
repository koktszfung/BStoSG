import torch.nn as nn


def get_bs2sg():
    layers = [
        nn.LeakyReLU(),
        nn.Linear(360, 100),
        nn.LeakyReLU(),
        nn.Linear(100, 100),
        nn.LeakyReLU(),
        nn.Linear(100, 230),
        nn.LeakyReLU(),
    ]
    model = nn.Sequential(*layers)
    return model


def get_bs2cs():
    layers = [
        nn.LeakyReLU(),
        nn.Linear(360, 200),
        nn.LeakyReLU(),
        nn.Linear(200, 100),
        nn.LeakyReLU(),
        nn.Linear(100, 7)
    ]
    model = nn.Sequential(*layers)
    return model


def get_cs2sg(*args):
    layers = []
    for i in range(len(args) - 1):
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(args[i], args[i+1]))
    model = nn.Sequential(*layers)
    return model
