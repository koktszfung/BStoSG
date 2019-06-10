import json
import numpy as np
import matplotlib.pyplot as plt


def plot_band_json(name):
    with open(name, "r") as file:
        data = json.load(file)
        x = np.empty(109)
        y = np.empty(109)
        for i, (key, val) in enumerate(data.items()):
            x[i] = key
            y[i] = val
        plt.bar(x, y)
        plt.show()


def plot_spacegroup(name):
    with open(name, "r") as file:
        x = np.arange(230)
        y = np.zeros(230)
        for i, line in enumerate(file):
            y[i] = line.split()[1]
        plt.bar(x, y)
        plt.show()


# id num_bands num_sites space_group
if __name__ == "__main__":
    plot_band_json("data/bands_res.json")
    plot_spacegroup("data/spacegroup_occurence_1-500.txt")
