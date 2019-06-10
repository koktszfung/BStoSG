import torch
import torch.nn
import torch.nn.functional
import torch.optim.adam
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
from scipy.special import softmax


class Set(Dataset):
    def __init__(self, data_dir, search_start, search_end, num_data):  # length: number of data
        self.len = num_data
        self.data_input = []
        self.data_label = []
        for subdir, dirs, files in os.walk(data_dir):  # search through the directory
            search_size = abs(search_end - search_start)  # separate input_data into test set and train set
            order = np.random.permutation(search_size) - 1 + search_start  # randomly add to set without repetition
            for i in range(search_size):  # iterate through order
                file_name = files[order[i]]
                with open(data_dir + file_name) as file:
                    data_json = json.load(file)
                    data_input_np = np.array(data_json["bands"])  # load bands into nd-array
                    if data_input_np.shape[0] != 30:  # accept only data with 30 energy bands
                        continue
                    data_input_np = softmax(data_input_np)
                    # data_input_np_max = np.max(np.abs(data_input_np))
                    # data_input_np = data_input_np / data_input_np_max
                    data_input_np = data_input_np.flatten().T
                    data_label_np = np.array([data_json["number"] - 1])
                self.data_input.append(torch.from_numpy(data_input_np).float())
                self.data_label.append(torch.from_numpy(data_label_np).long())
                if len(self.data_input) >= num_data:
                    break

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data_input[index], self.data_label[index]


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
        x = self.fc3(x)
        return x.view(1, 230)


def train(device: torch.device, network: torch.nn.Module, criterion, optimizer,
          data_loader: DataLoader, epoch_per_run):
    batch_per_loader = len(data_loader)
    item_per_batch = data_loader.batch_size
    for e in range(epoch_per_run):
        for b, batch in enumerate(data_loader):
            batch_input, batch_label = batch
            for i in range(item_per_batch):
                # reset gradient history
                optimizer.zero_grad()  # zero the gradient buffers
                # read data
                data_input, data_label = batch_input[i], batch_label[i]
                data_input, data_label = data_input.to(device), data_label.to(device)
                # feed
                output = network(data_input)
                # calculate loss
                cur_loss = criterion(output, data_label)
                # optimize
                cur_loss.backward()  # backpropagate and store changes needed
                optimizer.step()  # update weight and bias

                # progress bar
                if i % int(item_per_batch / 10) == 0:
                    print("\rtrain: epoch:{}/{} batch:{}/{} item:{}/{}".format(e, epoch_per_run, b, batch_per_loader, i,
                                                                               item_per_batch), end="")
            print("\rtrain: epoch:{}/{} batch:{}/{}".format(e, epoch_per_run, b, batch_per_loader), end="")
        print("\rtrain: epoch:{}/{}".format(e, epoch_per_run), end="")
    print("\rtrain: finished")


def test_multiple(device: torch.device, network: torch.nn.Module, criterion,
                  data_loader: DataLoader):
    batch_per_loader = len(data_loader)
    item_per_batch = data_loader.batch_size
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for b, batch in enumerate(data_loader):
            batch_input, batch_label = batch  # current batch
            for i in range(item_per_batch):
                # read data
                data_input, data_label = batch_input[i], batch_label[i]
                data_input, data_label = data_input.to(device), data_label.to(device)
                # feed
                output = network(data_input)
                # calculate loss
                cur_loss = criterion(output, data_label)
                # calculate fitness
                total_loss += cur_loss.item()
                if torch.max(output, 1)[1] == data_label:
                    total_correct += 1

                # progress bar
                if i % int(item_per_batch / 10) == 0:
                    print("\rtest: batch:{}/{} item:{}/{}".format(b, batch_per_loader, i, item_per_batch), end="")
            print("\rtest: batch:{}/{}".format(b, batch_per_loader), end="")
        print("\rtest: finished")
    num_turn = batch_per_loader * item_per_batch
    return total_loss / num_turn, total_correct/num_turn*100


def test_single(device: torch.device, network: torch.nn.Module, file_path):
    try:
        with open(file_path) as file:
            data_json = json.load(file)
            data_input_np = np.array(data_json["bands"])  # load bands into nd-array
            if data_input_np.shape[0] != 30:  # accept only data with 30 energy bands
                raise ValueError

            data_input_np = data_input_np.flatten().T
            data_label_np = data_json["number"] - 1

            data_input = torch.from_numpy(data_input_np).float().to(device)
            output = network(data_input)
            softmax_opt = torch.nn.Softmax(1)
            return data_label_np, (torch.round(softmax_opt(output)*100)/100).reshape(-1, 10)
    except ValueError:
        print("wrong number of bands")
        return None


def main():
    # setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    state_dict_path = ""
    if state_dict_path != "":
        net.load_state_dict(torch.load(state_dict_path))
        net.eval()
    net = net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    train_loader = DataLoader(dataset=Set("new_input_data/", 0, 9000, 5000), batch_size=1250, shuffle=True)
    test_loader = DataLoader(dataset=Set("new_input_data/", 9000, 11018, 500), batch_size=100, shuffle=True)

    result = test_multiple(device, net, criterion, test_loader)
    print("result:", result)

    train(device, net, criterion, optimizer, train_loader, 5)
    torch.save(net.state_dict(), "state_dict_save/temp.pt")

    result = test_multiple(device, net, criterion, test_loader)
    print("result:", result)

    results = test_single(device, net, "input_data/input_data_13139.json")
    for result in results:
        print(result, "\n")


def test():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    arr_max = np.amax(arr, 0)
    print(arr_max)
    print(arr/arr_max)


if __name__ == "__main__":
    main()
    # test()
    torch.cuda.empty_cache()
