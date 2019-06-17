import torch
import torch.nn as nn


def get_base_model():
    layers = []
    layers.append(nn.Softmax(0))
    layers.append(nn.Linear(360, 347))
    layers.append(nn.Tanh())
    layers.append(nn.Linear(347, 284))
    layers.append(nn.SELU())
    layers.append(nn.Linear(284, 230))
    layers.append(nn.SELU())
    model = nn.Sequential(*layers)
    return model
