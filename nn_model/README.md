# Bandstructure to Spacegroup
### [main.py](main.py)
- define `data_dir`, `optimizer`, `criterion`, `batch_size`, `train-valid_ratio`
- `validate`, `train`, `validate`
### [model.py](model.py)
- define network structure with `torch.nn.Module`
### [data_loader.py](data_loader.py)
- define `dataset` and `data_loader`
- load data from raw files into `data_loader`
### [filter_data.py](filter_data.py)
- provide the function `create_list_file`
- filter the data and print valid name into file `valid_name_list.txt`
