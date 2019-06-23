import json
import numpy


def count_in_guess():
    latnums = numpy.empty(7)
    for i in range(7):
        latnums[i] = len(open("../nn_model/data/crystal_{}_list.txt".format(i)).readlines())
    return latnums


def count_in_theory():
    crystals = numpy.zeros(7)
    for i in range(7):
        with open("../nn_model/data/crystal_{}_list.txt".format(i), "r") as list_file:
            for data_file_path in list_file:
                with open("../nn_model/" + data_file_path.split()[0], "r") as data_file:
                    data_json = json.load(data_file)
                    for crystal, sg_margins in enumerate([2, 15, 74, 142, 167, 194, 230]):
                        if data_json["number"] <= sg_margins:
                            crystals[crystal] += 1
                            break
    return crystals


def correct_count():
    corrects = numpy.zeros(7)
    for i in range(7):
        with open("../nn_model/data/crystal_{}_list.txt".format(i), "r") as list_file:
            for data_file_path in list_file:
                with open("../nn_model/" + data_file_path.split()[0], "r") as data_file:
                    data_json = json.load(data_file)
                    for crystal, sg_margins in enumerate([2, 15, 74, 142, 167, 194, 230]):
                        if data_json["number"] <= sg_margins and crystal == i:
                            corrects[i] += 1
                            break
    return corrects


guess = count_in_guess()
theory = count_in_theory()
correct = correct_count()
print(guess.sum())
print(theory.sum())
print(correct.sum())
