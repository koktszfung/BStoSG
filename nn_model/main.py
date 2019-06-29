import torch
import torch.nn
import torch.nn.functional

import data_processing
import base_models
import data_loaders
import neural_network


def main_bs_sg(num_epoch: int = 1, seed: int = 0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = base_models.get_bs2sg()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
    criterion = torch.nn.CrossEntropyLoss()

    dataset = data_loaders.SetBs2Sg("data/actual/valid_list.txt")
    train_loader, valid_loader = data_loaders.get_train_valid_loader(dataset, 32, 0.1, seed=seed)

    neural_network.validate_train_loop(
        device, model, optimizer, scheduler, criterion, valid_loader, train_loader, num_epoch
    )

    data_processing.create_guess_list_files(
        device, model, 230, "data/actual/valid_list.txt", "data/guess/", "spacegroup_list_{}.txt"
    )


def main_bs_crys(num_epoch: int = 1, seed: int = 0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = base_models.get_bs2crys()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
    criterion = torch.nn.CrossEntropyLoss()

    dataset = data_loaders.SetBs2Crys("data/actual/valid_list.txt")
    train_loader, valid_loader = data_loaders.get_train_valid_loader(dataset, 32, 0.1, seed=seed)

    neural_network.validate_train_loop(
        device, model, optimizer, scheduler, criterion, valid_loader, train_loader, num_epoch
    )

    data_processing.create_guess_list_files(
        device, model, 7, "data/actual/valid_list.txt", "data/guess/", "crystal_list_{}.txt"
    )


def main_crys_sg(crysnum: int, num_epoch: int = 1, seed: int = 0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    margins = [2, 15, 74, 142, 167, 194, 230]
    model = base_models.get_crys2sg(
        360, 100, 100, margins[crysnum] - margins[crysnum - 1] + 1 if crysnum > 0 else 3
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
    criterion = torch.nn.CrossEntropyLoss()

    dataset = data_loaders.SetCrys2Sg("data/guess/", crysnum)
    train_loader, valid_loader = data_loaders.get_train_valid_loader(
        dataset, 32, 0.1, seed=seed
    )

    neural_network.validate_train_loop(
        device, model, optimizer, scheduler, criterion, valid_loader, train_loader, num_epoch
    )

    data_processing.create_guess_list_files(
        device, model, 230, "data/actual/valid_list.txt", "data/guess/", "spacegroup_list_{}.txt"
    )


if __name__ == "__main__":
    torch.manual_seed(1155110044)
    data_processing.create_valid_list_files(30, "data/new_input_data_2/", "data/actual/valid_list.txt")

    main_bs_crys(10, 1155110044)

    torch.cuda.empty_cache()
    import winsound
    duration = 500  # milliseconds
    freq = 200  # Hz
    winsound.Beep(freq, duration)
