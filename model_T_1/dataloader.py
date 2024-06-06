from utils import Generate_Data, normalize_values
from torch.utils.data import DataLoader
import xarray as xr


def load_data(path, block_length = 8):
    data = xr.open_zarr(path)
    data_u10 = data['10m_u_component_of_wind']
    train_period = ["2019-01-01T00:00:00.000000000","2020-01-01T23:00:00.000000000"]
    validation_period = ["2020-02-01T00:00:00.000000000","2020-02-28T23:00:00.000000000"]
    testing_period = ["2020-03-01T00:00:00.000000000","2020-03-31T23:00:00.000000000"]

    longitude_range = [229.49999999999997, 299.99999999999994]
    latitude_range = [11.999999999999991, 70.49999999999999]

    X_train = data_u10.sel(time = slice(*train_period))
    X_train = X_train.sel(latitude = slice(*latitude_range))
    X_train = X_train.sel(longitude = slice(*longitude_range))
    X_train_norm = normalize_values(X_train)

    X_val = data_u10.sel(time = slice(*validation_period))
    X_val = X_val.sel(latitude = slice(*latitude_range))
    X_val = X_val.sel(longitude = slice(*longitude_range))
    X_val_norm = normalize_values(X_val)

    X_test = data_u10.sel(time = slice(*testing_period))
    X_test = X_test.sel(latitude = slice(*latitude_range))
    X_test = X_test.sel(longitude = slice(*longitude_range))
    X_test_norm = normalize_values(X_test)

    # return X_train_norm, X_val_norm, X_test_norm

    train_dataset = Generate_Data(X_train_norm, block_length)
    train_loader = DataLoader(train_dataset,batch_size = 4)

    val_dataset = Generate_Data(X_val_norm, block_length)
    val_loader = DataLoader(val_dataset,batch_size = 4)

    test_dataset = Generate_Data(X_test_norm, block_length)
    test_loader = DataLoader(test_dataset, batch_size = 4)
    
    datasets = {"train_dataset": train_dataset, "val_dataset":val_dataset, "test_dataset": test_dataset}
    dataloaders = {"train_dataloader": train_loader, "val_dataloader":val_loader, "test_dataloader": test_loader}
    
    return datasets, dataloaders