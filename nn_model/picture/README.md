# Network Setting Result
### old setting
- reduce input information by replacing energy degeneracies by that degree of degeneracies
![old input data](old_input_data)

### new setting
- keep the original energy level
![new input data](old_input_data)
- ratio only: all the entry is divided by the maximum absolute value
![new input data ratio only](new_input_data_ratio-only(all))
	- axis is the range for the maximum (0 for column, 1 for row)
	![new input data ratio only](new_input_data_ratio-only(axis=0))
	![new input data ratio only](new_input_data_ratio-only(axis=1))
- softmax: similar to sigmoid but result sums up to 1
![new input data softmax](new_input_data_softmax(all))
	- axis is the range for the normalization (0 for column, 1 for row)
	![new input data softmax](new_input_data_softmax(axis=0))
	![new input data softmax](new_input_data_softmax(axis=1))
