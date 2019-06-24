import json
import numpy


# DONE
def count_in_guess():
    latnums = numpy.empty(7)
    for i in range(7):
        latnums[i] = len(open("../nn_model/data/crystal_{}_list.txt".format(i)).readlines())
    return latnums


# DONE
def count_in_theory():
    crystals = numpy.zeros(7)
    for i in range(7):
        with open("../nn_model/data/crystal_{}_list.txt".format(i), "r") as list_file:
            for data_file_path in list_file:
                with open("../nn_model/" + data_file_path.split()[0], "r") as data_file:
                    data_json = json.load(data_file)

                    for c, margin in enumerate([2, 15, 74, 142, 167, 194, 230]):
                        if data_json["number"] <= margin:
                            crystals[c] += 1
                            break
    return crystals


def crystal_number(sgnum: int):
    for c, margin in enumerate([2, 15, 74, 142, 167, 194, 230]):
        if sgnum <= margin:
            return c


def correct_in_guess():
    corrects = numpy.zeros(7)
    for i in range(7):
        with open("../nn_model/data/crystal_{}_list.txt".format(i), "r") as list_file:
            for data_file_path in list_file:
                with open("../nn_model/" + data_file_path.split()[0], "r") as data_file:
                    data_json = json.load(data_file)
                    if crystal_number(data_json["number"]) == i:
                        corrects[i] += 1

    return corrects


guess_count = count_in_guess()
theory_count = count_in_theory()
guess_correct = correct_in_guess()
print("guess count: ", guess_count, guess_count.sum())
print("theory count: ", theory_count, theory_count.sum())
print("guess correct: ", guess_correct, guess_correct.sum())

print("correct percentage: ", (1 - (guess_count - guess_correct).sum()/guess_count.sum())*100)

print(guess_count - guess_correct)
print(theory_count - guess_correct)
