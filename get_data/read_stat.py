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
    spacegroup = np.loadtxt(name)
    plt.bar(spacegroup[:, 0], spacegroup[:, 1])
    plt.show()


if __name__ == "__main__":
    plot_band_json("data/bands_res.json")
    plot_spacegroup("data/spacegroup_weights.txt")
