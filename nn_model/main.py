import torch
import torch.nn
import torch.nn.functional
import numpy as np
import json

from filter_data import create_valid_list_files
import model_bs_sg
import model_bs_crys
import model_crys_sg
import data_loader_bs_sg
import data_loader_bs_crys
import data_loader_crys_sg


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


def main_bs_sg():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model_bs_sg.get_base_model()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    criterion = torch.nn.CrossEntropyLoss()

    create_valid_list_files("data/new_input_data_2/", 30)
    train_loader, valid_loader = data_loader_bs_sg.get_train_valid_loader("data/valid_list.txt", 32, 0.1)

    result = validate_one_epoch(device, model, criterion, valid_loader)
    print("\rvalid loss:{} accuracy:{}%".format(*result))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
    for epoch in range(10):
        result = train_one_epoch(device, model, optimizer, criterion, train_loader)
        print("\rtrain epoch:{} loss:{}".format(epoch, result))
        result = validate_one_epoch(device, model, criterion, valid_loader)
        print("\rvalid loss:{} accuracy:{}%".format(*result))
        scheduler.step(epoch)


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


def main_bs_crys():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model_bs_crys.get_base_model()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    criterion = torch.nn.CrossEntropyLoss()

    create_valid_list_files("data/new_input_data_2/", 30)
    train_loader, valid_loader = data_loader_bs_crys.get_train_valid_loader("data/valid_list.txt", 32, 0.1)

    result = validate_one_epoch(device, model, criterion, valid_loader)
    print("\rvalid loss:{} accuracy:{}%".format(*result))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
    for epoch in range(10):
        result = train_one_epoch(device, model, optimizer, criterion, train_loader)
        print("\rtrain epoch:{} loss:{}".format(epoch, result))
        result = validate_one_epoch(device, model, criterion, valid_loader)
        print("\rvalid loss:{} accuracy:{}%".format(*result))
        scheduler.step(epoch)

    # create_crystal_list_files(device, model, "data/valid_list.txt")


def main_crys_sg(crysnum: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    margins = [2, 15, 74, 142, 167, 194, 230]
    model = model_crys_sg.get_base_model(
        360, 100, 100, margins[crysnum] - margins[crysnum - 1] + 1 if crysnum > 0 else 3
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    criterion = torch.nn.CrossEntropyLoss()

    train_loader, valid_loader = data_loader_crys_sg.get_train_valid_loader(
        "data/", crysnum, 32, 0.1
    )

    result = validate_one_epoch(device, model, criterion, valid_loader)
    print("\rvalid crystal:{} loss:{} accuracy:{}%".format(crysnum, *result))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
    for epoch in range(10):
        result = train_one_epoch(device, model, optimizer, criterion, train_loader)
        print("\rtrain crystal:{} epoch:{} loss:{}".format(crysnum, epoch, result))
        result = validate_one_epoch(device, model, criterion, valid_loader)
        print("\rvalid crystal:{} loss:{} accuracy:{}%".format(crysnum, *result))
        scheduler.step(epoch)


if __name__ == "__main__":
    # main_bs_sg()
    # main_bs_crys()
    for i in range(7):
        main_crys_sg(i)

    torch.cuda.empty_cache()
    import winsound
    duration = 500  # milliseconds
    freq = 200  # Hz
    winsound.Beep(freq, duration)
