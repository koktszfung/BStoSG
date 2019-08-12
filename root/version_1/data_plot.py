import json
import numpy
from matplotlib import pyplot
import pandas
import seaborn

file_name_arr = numpy.loadtxt("data/actual/crystal_list_7.txt", "U60", ndmin=1)
cur_len = file_name_arr.size
for i in range(cur_len):
    file_name = file_name_arr[i]
    with open(file_name, "r") as file:
        data_json = json.load(file)
        data_input_np = numpy.array(data_json["bands"])

    dataframe_colormap = pandas.DataFrame(data_input_np)
    seaborn.heatmap(dataframe_colormap)
    pyplot.show()
