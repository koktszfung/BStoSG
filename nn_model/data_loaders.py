import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import json
import numpy


class SetBs2Sg(Dataset):
    def __init__(self, list_path):
        self.data_input = []
        self.data_label = []
        file_name_arr = numpy.loadtxt(list_path, "U50")
        self.len = len(file_name_arr)
        for i in range(self.len):
            file_name = file_name_arr[i]
            with open(file_name, "r") as file:
                data_json = json.load(file)
                data_input_np = numpy.array(data_json["bands"])
                data_input_np = data_input_np.flatten().T
                data_label_np = numpy.array([data_json["number"] - 1])
            self.data_input.append(torch.from_numpy(data_input_np).float())
            self.data_label.append(torch.from_numpy(data_label_np).long())
            print("\t\rload: {}/{}".format(i, self.len), end="")
        print("\rload: {}".format(self.len))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data_input[index], self.data_label[index]


class SetBs2Crys(Dataset):
    def __init__(self, list_path):
        self.data_input = []
        self.data_label = []
        file_name_arr = numpy.loadtxt(list_path, "U50")
        self.len = len(file_name_arr)
        for i in range(self.len):
            file_name = file_name_arr[i]
            with open(file_name, "r") as file:
                data_json = json.load(file)
                data_input_np = numpy.array(data_json["bands"])
                data_input_np = data_input_np.flatten().T
                for crysnum, margin in enumerate([2, 15, 74, 142, 167, 194, 230]):
                    if data_json["number"] <= margin:
                        data_label_np = numpy.array([crysnum])
                        break
            self.data_input.append(torch.from_numpy(data_input_np).float())
            self.data_label.append(torch.from_numpy(data_label_np).long())
            print("\t\rload: {}/{}".format(i, self.len), end="")
        print("\rload: {}".format(self.len))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data_input[index], self.data_label[index]


class SetCrys2Sg(Dataset):
    def __init__(self, list_dir, crysnum):
        self.data_input = []
        self.data_label = []
        list_path = list_dir + "crystal_list_{}.txt".format(crysnum)
        file_name_arr = numpy.loadtxt(list_path, "U50")
        self.len = len(file_name_arr)
        import random
        for i in range(self.len):
            file_name = file_name_arr[i]
            with open(file_name, "r") as file:
                data_json = json.load(file)
                data_input_np = numpy.array(data_json["bands"])
                data_input_np = data_input_np.flatten().T

                margins = [2, 15, 74, 142, 167, 194, 230]
                crystal_upper = margins[crysnum - 1]
                crystal_lower = margins[crysnum - 2] if crysnum > 1 else 0
                crystal_size = crystal_upper - crystal_lower

                data_label_val = data_json["number"] - crystal_lower - 1

                if data_label_val not in range(crystal_size):
                    data_label_val = random.randint(0, crystal_size)

                data_label_np = numpy.array([data_label_val])
            self.data_input.append(torch.from_numpy(data_input_np).float())
            self.data_label.append(torch.from_numpy(data_label_np).long())
            print("\t\rload: {}/{}".format(i, self.len), end="")
        print("\rload: {}".format(self.len))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data_input[index], self.data_label[index]


def get_train_valid_loader(dataset,
                           batch_size,
                           valid_size=0.1,
                           shuffle=True,
                           seed=0):
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(numpy.floor(valid_size * num_train))

    if shuffle:
        numpy.random.seed(seed)
        numpy.random.shuffle(indices)

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
