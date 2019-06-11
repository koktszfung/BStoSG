import os
import json
import numpy as np

for subdir, dirs, files in os.walk("data/new_input_data_2/"):
    with open("valid_name_list.txt", "w") as file_out:
        for file_name in files:
            with open("data/new_input_data_2/" + file_name, "r") as file_data:
                data_json = json.load(file_data)
                data_input_np = np.array(data_json["bands"])  # load bands into nd-array
                if data_input_np.shape[0] != 30:  # accept only data with 30 energy bands
                    continue
                file_out.writelines(file_name + "\n")
