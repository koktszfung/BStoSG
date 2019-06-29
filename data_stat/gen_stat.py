import numpy


def crystal_number(sgnum: int):
    for c, margin in enumerate([2, 15, 74, 142, 167, 194, 230]):
        if sgnum <= margin:
            return c + 1


def total_count(num_group, list_dir, list_format):
    counts = numpy.zeros(num_group).astype(int)
    for i in range(num_group):
        counts[i] = len(open(list_dir + list_format.format(i+1)).readlines())
    return counts


def correct_count(num_group, guess_list_dir, actual_list_dir, list_format):
    counts = numpy.zeros(num_group).astype(int)
    for i in range(num_group):
        with open(guess_list_dir + list_format.format(i+1), "r") as list_file:
            guesses = set([line.split("/")[-1] for line in list_file.readlines()])
        with open(actual_list_dir + list_format.format(i+1), "r") as list_file:
            actuals = set([line.split("/")[-1] for line in list_file.readlines()])
        counts[i] = len(set.intersection(guesses, actuals))
    return counts


def crystal_stat():
    guess_total = total_count(7, "../nn_model/data/guess/", "crystal_list_{}.txt")
    actual_total = total_count(7, "../nn_model/data/actual/", "crystal_list_{}.txt")
    guess_correct = correct_count(7, "../nn_model/data/guess/", "../nn_model/data/actual/", "crystal_list_{}.txt")
    print("guess count:", guess_total, guess_total.sum())
    print("actual count:", actual_total, actual_total.sum())
    print("guess correct:", guess_correct, guess_correct.sum())

    print("correct percentage: ", (1 - (guess_total - guess_correct).sum()/guess_total.sum())*100)

    print("TP:", guess_correct)
    print("TN", numpy.full(7, actual_total.sum()) - guess_total - actual_total + guess_correct)
    print("FP:", guess_total - guess_correct)
    print("FN:", actual_total - guess_correct)


def spacegroup_stat():
    guess_total = total_count(230, "../nn_model/data/guess/", "spacegroup_list_{}.txt")
    actual_total = total_count(230, "../nn_model/data/actual/", "spacegroup_list_{}.txt")
    guess_correct = correct_count(230, "../nn_model/data/guess/", "../nn_model/data/actual/", "spacegroup_list_{}.txt")
    print("guess count:", guess_total.sum(), "\n", guess_total)
    print("actual count:", actual_total.sum(), "\n", actual_total)
    print("guess correct:", guess_correct.sum(), "\n", guess_correct)

    print("correct percentage: ", (1 - (guess_total - guess_correct).sum()/guess_total.sum())*100)

    print("TP:\n", guess_correct)
    print("TN\n", numpy.full(230, actual_total.sum()) - guess_total - actual_total + guess_correct)
    print("FP:\n", guess_total - guess_correct)
    print("FN:\n", actual_total - guess_correct)


if __name__ == '__main__':
    crystal_stat()
    # spacegroup_stat()
    pass
