import torch
import torch.nn
import torch.nn.functional
import numpy as np

from model import get_base_model
from data_loader import get_train_valid_loader
from filter_data import create_list_file


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
        print("\rtrain batch:{}/{}".format(b, len(train_loader)), end="")
    return round(loss_epoch, 4)


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
        print("\rvalid batch:{}/{}".format(b, len(valid_loader)), end="")

    val_loss /= num_valid
    num_correct /= num_valid
    return round(val_loss, 4), round(num_correct*100, 4)


def main():
    np.set_printoptions(precision=2)
    # setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net = Net()
    model = get_base_model()
    model_path = ""  # "model_save/06102327_ce_softmax_all.pt"
    state_dict_path = ""  # "state_dict_save/06102327_ce_softmax_all.pt"

    if model_path != "":
        torch.load(model_path)
        model.eval()
    if state_dict_path != "":
        model.load_state_dict(torch.load(state_dict_path))
        model.eval()
    net = model.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.00526)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.00756)

    criterion = torch.nn.CrossEntropyLoss()

    create_list_file("data/new_input_data_5/", 30)
    train_loader, valid_loader = get_train_valid_loader("data/valid_name_list.txt", 32, 0.1)

    result = validate_one_epoch(device, net, criterion, valid_loader)
    print("\rvalid loss:{} accuracy:{}%".format(*result))

    for epoch in range(4):
        result = train_one_epoch(device, net, optimizer, criterion, train_loader)
        print("\rtrain epoch:{} loss:{}".format(epoch, result))

    result = validate_one_epoch(device, net, criterion, valid_loader)
    print("\rvalid loss:{} accuracy:{}%".format(*result))


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
    import winsound
    duration = 500  # milliseconds
    freq = 200  # Hz
    winsound.Beep(freq, duration)