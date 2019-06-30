import torch
import torch.nn
import torch.nn.functional

import data_processing
import base_models
import data_loaders
import neural_network


def main_bs2sg(num_epoch: int = 1, seed: int = 0):
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

    data_processing.create_empty_list_files(230, "data/guess/", "spacegroup_list_{}.txt")
    data_processing.create_guess_list_files(
        device, model, 230, "data/actual/valid_list.txt", "data/guess/", "spacegroup_list_{}.txt"
    )


def main_bs2crys(num_epoch: int = 1, seed: int = 0):
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


def main_crys2sg_one(crysnum: int, num_epoch: int = 1, seed: int = 0):
    print("crystal system: {}".format(crysnum))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    margins = [2, 15, 74, 142, 167, 194, 230]
    model = base_models.get_crys2sg(
        # 360, 100, 100, margins[crysnum - 1] - margins[crysnum - 2] + 1 if crysnum > 1 else 3
        360, 100, 100, margins[crysnum - 1] - margins[crysnum - 2] if crysnum > 1 else 2
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
    criterion = torch.nn.CrossEntropyLoss()

    dataset = data_loaders.SetCrys2Sg("data/actual/", crysnum)
    train_loader, valid_loader = data_loaders.get_train_valid_loader(
        dataset, 32, 0.1, seed=seed
    )

    neural_network.validate_train_loop(
        device, model, optimizer, scheduler, criterion, valid_loader, train_loader, num_epoch
    )

    data_processing.append_guess_spacegroup_in_crystal_list_files(
        device, model, crysnum, "data/guess/crystal_list_{}.txt".format(crysnum), "data/guess/"
    )


def main_crys2sg_all(num_epoch: int = 1, seed: int = 0):
    data_processing.create_empty_list_files(230, "data/guess/", "spacegroup_list_{}.txt")
    for i in range(1, 8):
        main_crys2sg_one(i, num_epoch, seed)


if __name__ == "__main__":
    my_seed = 1155110044
    torch.manual_seed(my_seed)

    # prepare actual data #
    # data_processing.create_valid_list_files(30, "data/new_input_data_2/", "data/actual/valid_list.txt")
    # data_processing.create_actual_crystal_list_files("data/actual/valid_list.txt", "data/actual/")
    # data_processing.create_actual_spacegroup_list_files("data/actual/valid_list.txt", "data/actual/")

    # generate guess data #
    # main_bs2sg(num_epoch=10, seed=my_seed)
    # main_bs2crys(num_epoch=10, seed=my_seed)
    # main_crys2sg_all(num_epoch=10, seed=my_seed)

    torch.cuda.empty_cache()
    import winsound
    duration = 500  # milliseconds
    freq = 200  # Hz
    winsound.Beep(freq, duration)
