import torch.nn as nn


def get_base_model():
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
