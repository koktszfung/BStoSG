import torch
import torch.nn
import torch.nn.functional
import torch.optim.adam
from torch.utils.data import Dataset, DataLoader
import os
import json
import math
import numpy as np
from scipy.special import softmax
from my_data_loader import get_train_valid_loader


class Net(torch.nn.Module):
    """
    default function: __init__, forward, backward, backward is auto generated
    """
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(360, 100)
        self.fc2 = torch.nn.Linear(100, 100)
        self.fc3 = torch.nn.Linear(100, 230)

    def forward(self, x):
        # rrelu: activation, self.fc(x): pass previous output as input through fc
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = torch.nn.functional.leaky_relu(self.fc3(x))
        # softmax_opt = torch.nn.Softmax(1)
        # return softmax_opt(x.view(1, 230))
        return x.view(1, 230)


def train_one_epoch(device, model, optimizer, criterion, train_loader):
    model.train()
    loss_epoch = 0.
    for b, batch in enumerate(train_loader):
        batch_input, batch_label = batch
        for i in range(len(batch[0])):
            # reset gradient history
            optimizer.zero_grad()  # zero the gradient buffers
            # read data
            data_input, data_label = batch_input[i], batch_label[i]
            data_input, data_label = data_input.to(device), data_label.to(device)
            # feed
            output = model(data_input).view(1, 230)
            loss = criterion(output, data_label)
            loss.backward()
            optimizer.step()
            loss_epoch = loss.item()
    return loss_epoch


def validate_one_epoch(device, model, criterion, valid_loader):
    model.eval()
    num_valid = len(valid_loader.sampler.indices)
    val_loss = 0.
    num_correct = 0
    for b, batch in enumerate(valid_loader):
        batch_input, batch_label = batch  # current batch
        for i in range(len(batch[0])):
            # read data
            data_input, data_label = batch_input[i], batch_label[i]
            data_input, data_label = data_input.to(device), data_label.to(device)
            output = model(data_input).view(1, 230)
            val_loss += criterion(output, data_label).item()

            if torch.max(output, 1)[1] == data_label:
                num_correct += 1

    val_loss /= num_valid
    num_correct /= num_valid
    return val_loss, num_correct


def main():
    np.set_printoptions(precision=2)
    # setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    model_path = ""  # "model_save/06102327_ce_softmax_all.pt"
    state_dict_path = ""  # "state_dict_save/06102327_ce_softmax_all.pt"

    if model_path != "":
        torch.load(model_path)
        net.eval()
    if state_dict_path != "":
        net.load_state_dict(torch.load(state_dict_path))
        net.eval()
    net = net.to(device)

    # optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.044)

    criterion = torch.nn.CrossEntropyLoss()

    train_loader, valid_loader = get_train_valid_loader("data/new_input_data_3/", 500, 0.3)

    result = validate_one_epoch(device, net, criterion, valid_loader)
    print("test: loss:{} num:{}".format(*result))

    for epoch in range(4):
        result = train_one_epoch(device, net, optimizer, criterion, train_loader)
        print("train: epoch:{} loss:{}".format(epoch, result))

    result = validate_one_epoch(device, net, criterion, valid_loader)
    print("test: loss:{} num:{}".format(*result))


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
