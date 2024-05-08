# %% explore data
path = 'DA_every10HR_lead3_everytime_noise_0.001.mat' 
from scipy.io import loadmat
# %%
mat_contents = loadmat(path)
# %%
prediction = mat_contents['prediction']
truth = mat_contents['truth']
noisy_obs = mat_contents['noisy_obs']

# print(mat_contents.keys())
# %%
print( prediction.shape, truth.shape, noisy_obs.shape)
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from dataloader import load_data

obs_path = f"gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr"
datasets, _ = load_data(obs_path)
test_dataset = datasets['test_dataset']
lats = test_dataset.data['latitude'].values #32
lons = test_dataset.data['longitude'].values #64

# %%
value = truth[0].squeeze()
print(value.T.shape)

model_path = "models/model_20240507_013640_99"
figures_path = f"Figures/{model_path[7:]}/" 

if not os.path.exists(figures_path):
    os.makedirs(figures_path)
# %% plots
#series of images:
for i in range(0,truth.shape[0]):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    # Plot for truth values
    axs[0].coastlines()
    lons_2d, lats_2d = np.meshgrid(lons, lats)
    sc1 = axs[0].pcolormesh(lons_2d, lats_2d, truth[i].squeeze().T, cmap='Spectral_r', vmin=0., vmax=1., transform=ccrs.PlateCarree())
    plt.colorbar(sc1, ax=axs[0], label='10m_u_component_of_wind')
    axs[0].set_title("Truth value")

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

# %%

# M, sdev = 0.47228286, 0.12093292
# import xarray as xr
# from utils import normalize_values
# obs_path = f"gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr"
# data = xr.open_zarr(obs_path)
# data_u10 = data['10m_u_component_of_wind']
# testing_period = ["2020-02-11T00:00:00.000000000","2020-02-20T23:00:00.000000000"]
# X_test = data_u10.sel(time = slice(*testing_period))
# X_test_norm = normalize_values(X_test).to_numpy()

# X_test = X_test.to_numpy()
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
# X_test_true = X_test
# X_test_true = (X_test_true - M)/sdev

# # %%
# print(X_test_norm.shape)

# # %%
# i = 0
# X_test_true = X_test_true.reshape(40,64,32)
# fig, axs = plt.subplots(1, 2, figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()})
# axs[0].coastlines()
# sc2 = axs[0].pcolormesh(lons_2d, lats_2d, X_test_true[i].squeeze().T, cmap='Spectral_r', vmin=0., vmax=1., transform=ccrs.PlateCarree())
# plt.colorbar(sc2, ax=axs[0], label='10m_u_component_of_wind')
# axs[0].set_title("Prediction value")

# axs[1].coastlines()
# sc2 = axs[1].pcolormesh(lons_2d, lats_2d, X_test_norm[i].squeeze().T, cmap='Spectral_r', vmin=0., vmax=1., transform=ccrs.PlateCarree())
# plt.colorbar(sc2, ax=axs[1], label='10m_u_component_of_wind')
# axs[1].set_title("Prediction value")
# %% values in truth
import xarray as xr
from utils import normalize_values

obs_path = f"gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr"
data = xr.open_zarr(obs_path)
data_u10 = data['10m_u_component_of_wind']
testing_period = ["2020-02-11T00:00:00.000000000","2020-02-20T23:00:00.000000000"]
X_test = data_u10.sel(time = slice(*testing_period))
X_test = normalize_values(X_test).to_numpy()
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
X_test_true = X_test
X_test_true = X_test_true.reshape([np.size(X_test,0),64,32,1])
# %% plotter

i = 0
fig, axs = plt.subplots(1, 2, figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()})
# Plot for truth values
axs[0].coastlines()
lons_2d, lats_2d = np.meshgrid(lons, lats)
sc1 = axs[0].pcolormesh(lons_2d, lats_2d, X_test_true[i].squeeze().T, cmap='Spectral_r', vmin=0., vmax=1., transform=ccrs.PlateCarree())
plt.colorbar(sc1, ax=axs[0], label='10m_u_component_of_wind')
axs[0].set_title("Truth value")

# Plot for prediction values    
axs[1].coastlines()
sc2 = axs[1].pcolormesh(lons_2d, lats_2d, prediction[i].squeeze().T, cmap='Spectral_r', vmin=0., vmax=1., transform=ccrs.PlateCarree())
plt.colorbar(sc2, ax=axs[1], label='10m_u_component_of_wind')
axs[1].set_title("Prediction value")
plt.tight_layout()
fig.suptitle(f"True value and Prediction for {test_dataset.data[i]['time'].values}, prediction = {i}")
plt.show()

# plt.savefig(f"{figures_path}/True value and Prediction for {test_dataset.data[i]['time'].values}_{model_path[7:]}.png")
# plt.close()
