import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import json
import numpy as np
from scipy.special import softmax


class Set(Dataset):
    def __init__(self, data_dir):
        self.data_input = []
        self.data_label = []
        file_name_arr = np.loadtxt("data/valid_name_list.txt", "U30")
        self.len = len(file_name_arr)
        for i in range(self.len):
            file_name = file_name_arr[i]
            with open(data_dir + file_name, "r") as file:
                data_json = json.load(file)
                data_input_np = np.array(data_json["bands"])  # load bands into nd-array.

                data_input_np = softmax(data_input_np)
                # data_input_np_max = np.max(np.abs(data_input_np), 0)
                # data_input_np = data_input_np / data_input_np_max

                data_input_np = data_input_np.flatten().T
                data_label_np = np.array([data_json["number"] - 1])
            self.data_input.append(torch.from_numpy(data_input_np).float())
            self.data_label.append(torch.from_numpy(data_label_np).long())

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data_input[index], self.data_label[index]


def get_train_valid_loader(data_dir,
                           batch_size,
                           valid_size=0.1,
                           shuffle=True):

    dataset = Set(data_dir)

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
