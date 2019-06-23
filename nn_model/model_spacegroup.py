import torch.nn as nn


def get_base_model():
    layers = [
        nn.Linear(360, 100),
        nn.LeakyReLU(),
        nn.Linear(100, 100),
        nn.LeakyReLU(),
        nn.Linear(100, 230)
    ]
    model = nn.Sequential(*layers)
    return model
