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
    def __init__(self, data_dir, num_data, start=0, end=20045):
        self.len = num_data
        self.data_input = []
        self.data_label = []
        search_size = abs(end - start)  # separate input_data into test set and train set
        order = np.random.permutation(search_size) - 1 + start  # randomly add to set without repetition
        file_name_arr = np.loadtxt("valid_name_list.txt", "U30")
        for i in range(search_size):
            file_name = file_name_arr[order[i]]
            with open(data_dir + file_name, "r") as file:
                data_json = json.load(file)
                data_input_np = np.array(data_json["bands"])  # load bands into nd-array.
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
        x = torch.nn.functional.leaky_relu(self.fc3(x))
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
                if torch.max(output, 1)[1] == data_label:  # torch.max(tensor, axis) -> (value, index)
                    total_correct += 1

                # progress bar
                if i % int(item_per_batch / 10) == 0:
                    print("\rtest: batch:{}/{} item:{}/{}".format(b, batch_per_loader, i, item_per_batch), end="")
            print("\rtest: batch:{}/{}".format(b, batch_per_loader), end="")
        print("\rtest: finished")
    num_turn = batch_per_loader * item_per_batch
    return total_loss / num_turn, total_correct/num_turn*100


def test_show_result(device: torch.device, network: torch.nn.Module, data_loader: DataLoader):
    with torch.no_grad():
        data_input, data_label = data_loader.dataset.__getitem__(0)
        data_input, data_label = data_input.to(device), data_label.to(device)
        output = network(data_input)
        softmax_opt = torch.nn.Softmax(1)
        return data_label.item(), torch.max(output, 1)[1].item(), (torch.round(softmax_opt(output)*100)/100).reshape(-1, 10)


def test_specific(device: torch.device, network: torch.nn.Module, file_path):
    with torch.no_grad():
        try:
            with open(file_path) as file:
                data_json = json.load(file)
                data_input_np = np.array(data_json["bands"])  # load bands into nd-array
                if data_input_np.shape[0] != 30:  # accept only data with 30 energy bands
                    raise ValueError
                data_input_np = softmax(data_input_np)
                # data_input_np_max = np.max(np.abs(data_input_np))
                # data_input_np = data_input_np / data_input_np_max
                data_input_np = data_input_np.flatten().T
                data_label_np = data_json["number"] - 1

                data_input = torch.from_numpy(data_input_np).float().to(device)
                output = network(data_input)
                softmax_opt = torch.nn.Softmax(1)
                return data_label_np, torch.max(output, 1)[1].item(), (torch.round(softmax_opt(output)*100)/100).reshape(-1, 10)
        except ValueError:
            print("wrong number of bands")
            return None


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

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    train_loader = DataLoader(dataset=Set("data/new_input_data_2/", 5000, end=14000), batch_size=1250, shuffle=True)
    test_loader = DataLoader(dataset=Set("data/new_input_data_2/", 3000, start=14000), batch_size=100, shuffle=True)

    result = test_multiple(device, net, criterion, test_loader)
    print("result:", result)

    train(device, net, criterion, optimizer, train_loader, 8)
    torch.save(net, "model_save/temp.pt")
    torch.save(net.state_dict(), "state_dict_save/temp.pt")

    result = test_multiple(device, net, criterion, test_loader)
    print("result:", result)

    results = test_specific(device, net, "data/new_input_data_2/new_input_data_13139.json")
    print("guess:{} index:{}".format(results[1], results[0]))

    for i in range(10):
        test_loader = DataLoader(dataset=Set("data/new_input_data_2/", 1, start=14000), shuffle=True)
        results = test_show_result(device, net, test_loader)
        print("guess:{} index:{}".format(results[1], results[0]))


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
