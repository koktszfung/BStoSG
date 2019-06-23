import json
import numpy


def get_sg_nums():
    with open("sg_weights.json", "r") as file:
        return numpy.array(json.load(file)["sg_weights"])


def get_lattice_nums_old():
    latnums = numpy.empty(7)
    sgnums = get_sg_nums()
    latnums[0] = sgnums[0:2, 1].sum()
    latnums[1] = sgnums[2:15, 1].sum()
    latnums[2] = sgnums[15:74, 1].sum()
    latnums[3] = sgnums[74:142, 1].sum()
    latnums[4] = sgnums[142:167, 1].sum()
    latnums[5] = sgnums[167:194, 1].sum()
    latnums[6] = sgnums[194:230, 1].sum()
    return latnums


def get_lattice_nums_guess():
    latnums = numpy.empty(7)
    for i in range(7):
        latnums[i] = len(open("../nn_model/data/lattice_{}_list.txt".format(i)).readlines())
    return latnums


def gen_lattice_nums():
    latnums = numpy.zeros(7)
    for i in range(7):
        with open("../nn_model/data/lattice_{}_list.txt".format(i), "r") as list_file:
            for data_file_path in list_file:
                with open("../nn_model/" + data_file_path.split()[0], "r") as data_file:
                    data_json = json.load(data_file)
                    for lattice, sg_margins in enumerate([2, 15, 74, 142, 167, 194, 230]):
                        if data_json["number"] <= sg_margins:
                            latnums[lattice] += 1
                            break
    return latnums


actual = gen_lattice_nums()
guess = get_lattice_nums_guess()
print(actual, actual.sum())
print(guess, guess.sum())
print(actual - guess)
print((1 - abs(actual - guess).sum()/2/actual.sum())*100)
