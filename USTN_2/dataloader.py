from utils import Generate_Data, normalize_values
from torch.utils.data import DataLoader
import xarray as xr


def load_data(path, in_len = 1, out_len = 1, lead = 3):
    data = xr.open_zarr(path)
    data_u10 = data['10m_u_component_of_wind']
    train_period = ["2020-01-01T00:00:00.000000000","2020-01-31T23:00:00.000000000"]
    validation_period = ["2020-02-01T00:00:00.000000000","2020-02-10T23:00:00.000000000"]
    testing_period = ["2020-02-11T00:00:00.000000000","2020-02-20T23:00:00.000000000"]

    X_train = data_u10.sel(time = slice(*train_period))
    X_train_norm = normalize_values(X_train)

    X_val = data_u10.sel(time = slice(*validation_period))
    X_val_norm = normalize_values(X_val)

    X_test = data_u10.sel(time = slice(*testing_period))
    X_test_norm = normalize_values(X_test)

    train_dataset = Generate_Data(X_train_norm, in_len, out_len, lead)
    train_loader = DataLoader(train_dataset,batch_size = 4)

    val_dataset = Generate_Data(X_val_norm, in_len, out_len, lead)
    val_loader = DataLoader(val_dataset,batch_size = 4)

    test_dataset = Generate_Data(X_test_norm, in_len, out_len, lead)
    test_loader = DataLoader(test_dataset, batch_size = 4)
    
    datasets = {"train_dataset": train_dataset, "val_dataset":val_dataset, "test_dataset": test_dataset}
    dataloaders = {"train_dataloader": train_loader, "val_dataloader":val_loader, "test_dataloader": test_loader}
    
    return datasets, dataloaders


    

