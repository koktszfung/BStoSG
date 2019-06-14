import os
import json
import numpy as np

for subdir, dirs, files in os.walk("results/"):
    with open("result_list.txt", "w") as file_out:
        for file_name in files:
            with open("results/" + file_name, "r") as file_data:
                data_json = json.load(file_data)
                lr = data_json["params"]["lr"]
                optim = data_json["params"]["optim"]
                val_loss = data_json["val_loss"]
                file_out.write("{}\n\tlr:{} optim:{} val loss:{}\n".format(
                    file_name[:-5], round(lr, 4), optim, round(val_loss, 4))
                )
