import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from dataloader import load_data
import xarray as xr
from utils import normalize_values
import os

obs_path = f"gs://weatherbench2/datasets/era5/1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr/"
data = xr.open_zarr(obs_path)
longitude_range = [229.49999999999997, 299.99999999999994]
latitude_range = [11.999999999999991, 70.49999999999999]

lats = data['latitude'].sel(latitude = slice(*latitude_range)) 
lons = data['longitude'].sel(longitude = slice(*longitude_range)) 

# lons = test_dataset.data['longitude'].values 
# X_train = X_train.sel(latitude = slice(*latitude_range))
# %% reading ENKF results
import scipy.io
mat = scipy.io.loadmat("DA_every10HR_lead3_everytime_noise_model_20240531_222522_99_0.001.mat")
prediction = mat['prediction']
truth = mat['truth']
noisy_obs = mat['noisy_obs']
n_steps = prediction.shape[0]

model_path = "models/model_20240531_222522_99"
figures_path = f"Figures/enkf_results/{model_path[7:]}" 
lons_2d, lats_2d = np.meshgrid(lons, lats)

if not os.path.exists(figures_path):
    os.makedirs(figures_path)

from dataloader import load_data
data_sets, _ = load_data(obs_path)
test_dataset = data_sets['test_dataset']


# print(lons_2d.shape, lats_2d.shape, truth[0].squeeze().shape)
for i in range(0,n_steps):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    # Plot for truth values
    axs[0].coastlines()
    sc1 = axs[0].pcolormesh(lons_2d, lats_2d, truth[i].squeeze().T, cmap='Spectral_r', vmin=0., vmax=1., transform=ccrs.PlateCarree())
    plt.colorbar(sc1, ax=axs[0], label='10m_u_component_of_wind')
    axs[0].set_title("Test value")

    # Plot for prediction values
    axs[1].coastlines()
    sc2 = axs[1].pcolormesh(lons_2d, lats_2d, prediction[i].squeeze().T, cmap='Spectral_r', vmin=0., vmax=1., transform=ccrs.PlateCarree())
    plt.colorbar(sc2, ax=axs[1], label='10m_u_component_of_wind')
    axs[1].set_title("Prediction value")
    
    plt.tight_layout()
    fig.suptitle(f"True value and Prediction for {test_dataset.data[i]['time'].values}, prediction = {i}")
    # plt.show()
    plt.savefig(f"{figures_path}/True value and Prediction for {test_dataset.data[i]['time'].values}_{model_path[7:]}.png")
    plt.close() 