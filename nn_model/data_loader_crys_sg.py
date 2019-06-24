import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import json
import numpy as np


class Set(Dataset):
    def __init__(self, list_dir, crysnum):
        self.data_input = []
        self.data_label = []
        list_path = list_dir + "crystal_{}_list.txt".format(crysnum)
        file_name_arr = np.loadtxt(list_path, "U50")
        self.len = len(file_name_arr)
        for i in range(self.len):
            file_name = file_name_arr[i]
            with open(file_name, "r") as file:
                data_json = json.load(file)
                data_input_np = np.array(data_json["bands"])
                data_input_np = data_input_np.flatten().T

                margins = [2, 15, 74, 142, 167, 194, 230]
                crystal_upper = margins[crysnum]
                crystal_lower = margins[crysnum - 1] if crysnum > 0 else 0
                crystal_size = crystal_upper - crystal_lower

                data_label_val = data_json["number"] - crystal_lower - 1
                data_label_val = data_label_val if data_label_val in range(crystal_size) else crystal_size

                data_label_np = np.array([data_label_val])
            self.data_input.append(torch.from_numpy(data_input_np).float())
            self.data_label.append(torch.from_numpy(data_label_np).long())
            print("\rload: {}/{}".format(i, self.len), end="")
        print("\rload: {}".format(self.len))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data_input[index], self.data_label[index]


def get_train_valid_loader(list_dir,
                           crysnum,
                           batch_size,
                           valid_size=0.1,
                           shuffle=True):

    dataset = Set(list_dir, crysnum)

    # create dataloaders
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        seed = 1155110044
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
    )
    valid_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
    )

    return train_loader, valid_loader
