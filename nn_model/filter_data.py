import os
import json
import numpy as np


def create_valid_list_files(data_dir, num_bands):
    for subdir, dirs, files in os.walk(data_dir):
        with open("data/valid_list.txt", "w") as file_out:
            for file_name in files:
                with open(data_dir + file_name, "r") as file_data:
                    data_json = json.load(file_data)
                    data_input_np = np.array(data_json["bands"])  # load bands into nd-array
                    if data_input_np.shape[0] != num_bands:  # accept only data with 30 energy bands
                        continue
                    file_out.write(data_dir + file_name + "\n")


if __name__ == "__main__":
    create_valid_list_files("data/new_input_data_2/", 30)
