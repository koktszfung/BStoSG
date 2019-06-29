import os
import json
import torch
import numpy


def create_valid_list_files(num_bands, in_data_dir, out_list_path):
    print("\tcreate valid list:", end="")
    with open(out_list_path, "w") as file_out:
        for subdir, dirs, files in os.walk(in_data_dir):
            for file_name in files:
                with open(in_data_dir + file_name, "r") as file_data:
                    data_json = json.load(file_data)
                    data_input_np = numpy.array(data_json["bands"])  # load bands into nd-array
                    if data_input_np.shape[0] != num_bands:  # accept only data with certain number of bands
                        continue
                    file_out.write(in_data_dir + file_name + "\n")
    print("\rcreate valid list: {}".format(len(open(out_list_path).readlines())))


def create_empty_list_files(out_num_group, out_list_dir, list_format):
    for i in range(out_num_group):
        open(out_list_dir + list_format.format(i+1), "w").close()


def create_actual_spacegroup_list_files(in_list_path, out_list_dir):
    file_paths = numpy.loadtxt(in_list_path, "U70")
    for file_path in file_paths:
        with open(file_path, "r") as file:
            data_json = json.load(file)
            sgnum = data_json["number"]
        with open(out_list_dir + "spacegroup_list_{}.txt".format(sgnum), "a") as file_out:
            file_out.write(file_path + "\n")


def create_actual_crystal_list_files(in_list_path, out_list_dir):
    def crystal_number(s: int):
        for c, margin in enumerate([2, 15, 74, 142, 167, 194, 230]):
            if s <= margin:
                return c + 1
    file_paths = numpy.loadtxt(in_list_path, "U70")
    for file_path in file_paths:
        with open(file_path, "r") as file:
            data_json = json.load(file)
            sgnum = data_json["number"]
            crysnum = crystal_number(sgnum)
        with open(out_list_dir + "crystal_list_{}.txt".format(crysnum), "a") as file_out:
            file_out.write(file_path + "\n")


def create_guess_list_files(device, model, in_list_path, out_list_dir, list_format):
    file_paths = numpy.loadtxt(in_list_path, "U70")
    for i, file_path in enumerate(file_paths):
        with open(file_path, "r") as file:
            data_json = json.load(file)
            data_input_np = numpy.array(data_json["bands"])
            data_input_np = data_input_np.flatten().T
            data_input = torch.from_numpy(data_input_np).float()
            output = model(data_input.to(device))
            number = torch.max(output, 0)[1].item() + 1
        with open(out_list_dir + list_format.format(number), "a") as file_out:
            file_out.write(file_path + "\n")
        print("\r\tcreate guess list: {}/{}".format(i, len(file_paths)), end="")
    print("\rcreate guess list: {}".format(len(file_paths)))


def create_guess_spacegroup_in_crystal_list_files(device, model, crysnum, in_list_path, out_list_dir):
    margins = [2, 15, 74, 142, 167, 194, 230]
    crystal_lower = margins[crysnum - 2] if crysnum > 1 else 0

    file_paths = numpy.loadtxt(in_list_path, "U70")
    for i, file_path in enumerate(file_paths):
        with open(file_path, "r") as file:
            data_json = json.load(file)
            data_input_np = numpy.array(data_json["bands"])
            data_input_np = data_input_np.flatten().T
            data_input = torch.from_numpy(data_input_np).float()
            output = model(data_input.to(device))
            sgnum = torch.max(output, 0)[1].item() + 1 + crystal_lower
        with open(out_list_dir + "spacegroup_list_{}.txt".format(sgnum), "a") as file_out:
            file_out.write(file_path + "\n")
        print("\r\tcreate guess list: {}/{}".format(i, len(file_paths)), end="")
    print("\rcreate guess list: {}".format(len(file_paths)))


if __name__ == "__main__":
    # create_valid_list_files(30, "../nn_model/data/new_input_data_2/", "../nn_model/data/actual/valid_list.txt")
    # create_actual_spacegroup_list_files("../nn_model/data/actual/valid_list.txt", "../nn_model/data/actual/")
    # create_actual_crystal_list_files("../nn_model/data/actual/valid_list.txt", "../nn_model/data/actual/")
    pass
