import torch
import torch.nn as nn


def get_base_model():
    layers = []
    layers.append(nn.LeakyReLU())
    layers.append(nn.Linear(360, 91))
    layers.append(nn.Sigmoid())
    layers.append(nn.Linear(91, 35))
    layers.append(nn.LeakyReLU())
    layers.append(nn.Linear(35, 230))
    layers.append(nn.SELU())
    model = nn.Sequential(*layers)
    return model
