import torch
import torch.nn
import torch.nn.functional
import numpy as np
import json

from model_crystal import get_base_model
from data_loader_crystal import get_train_valid_loader
from filter_data import create_valid_list_files


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
            output = model(data_input).view(1, -1)
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
            output = model(data_input).view(1, -1)
            val_loss += criterion(output, data_label).item()

            if torch.max(output, 1)[1] == data_label:
                num_correct += 1
        print("\rvalid batch:{}/{}".format(b, len(valid_loader)), end="")

    val_loss /= num_valid
    num_correct /= num_valid
    return round(val_loss, 4), round(num_correct*100, 4)


def create_crystal_list_files(device, model, list_path):
    file_name_arr = np.loadtxt(list_path, "U50")
    for i in range(7):
        open("data/crystal_{}_list.txt".format(i), "w").close()
    for i in range(file_name_arr.shape[0]):
        file_name = file_name_arr[i]
        with open(file_name, "r") as file:
            data_json = json.load(file)
        data_input_np = np.array(data_json["bands"])
        data_input_np = data_input_np.flatten().T
        data_input = torch.from_numpy(data_input_np).float()
        output = model(data_input.to(device))
        guess_crystal = torch.max(output, 0)[1].item()
        with open("data/crystal_{}_list.txt".format(guess_crystal), "a") as file:
            file.write(file_name + "\n")
        print("\rcreate crystal file: {}/{}".format(i, file_name_arr.shape[0]), end="")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_base_model()
    model_path = ""  # "model_save/06102327_ce_softmax_all.pt"
    state_dict_path = ""  # "state_dict_save/06102327_ce_softmax_all.pt"

    if model_path != "":
        torch.load(model_path)
        model.eval()
    if state_dict_path != "":
        model.load_state_dict(torch.load(state_dict_path))
        model.eval()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.00756)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)

    criterion = torch.nn.CrossEntropyLoss()

    create_valid_list_files("data/new_input_data_2/", 30)
    train_loader, valid_loader = get_train_valid_loader("data/valid_list.txt", 32, 0.1)

    result = validate_one_epoch(device, model, criterion, valid_loader)
    print("\rvalid loss:{} accuracy:{}%".format(*result))

    for epoch in range(4):
        result = train_one_epoch(device, model, optimizer, criterion, train_loader)
        print("\rtrain epoch:{} loss:{}".format(epoch, result))
        result = validate_one_epoch(device, model, criterion, valid_loader)
        print("\rvalid loss:{} accuracy:{}%".format(*result))
        scheduler.step(epoch)

    create_crystal_list_files(device, model, "data/valid_list.txt")


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
    import winsound
    duration = 500  # milliseconds
    freq = 200  # Hz
    winsound.Beep(freq, duration)
