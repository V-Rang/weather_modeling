# %% rolling forecast set-up
from model import Transformer_model
from dataloader import load_data
import torch
from tqdm import tqdm
import os
import numpy as np
block_size = 8
batch_size = 4

model_loaded = Transformer_model(48*40)
model_path = "models/model_20240606_054629_9"
model_loaded.load_state_dict(torch.load(model_path))

obs_path = f"gs://weatherbench2/datasets/era5/1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr/"
data_sets, data_loaders = load_data(obs_path)

test_loader = data_loaders['test_dataloader'] 
test_dataset = data_sets['test_dataset'] 

#implementing rolling forecast
# (0) -> (1) | compare to (1) of actual
# (1) -> (2) | compare to (2) of actual
# ...
def rolling_forecast(model, Xt, n_steps):
    # Initialize states
    y_pred = Xt
    # Iterate over timesteps
    # for k in range(n_steps):
    #     ym = torch.tensor(y_pred[-block_size:].reshape(1,*y_pred[-block_size:].shape))
    #     yp = model(ym)
    #     y_pred = np.concatenate((y_pred, yp.detach().numpy().squeeze(axis = 0)), axis=0)        
    # return y_pred[Xt.shape[0] :]
    y_pred = X_init
    n_steps = len(test_dataset.data) - 1
    for _ in range(n_steps):
        ym = torch.tensor(y_pred[:,-block_size:])
        yp = model(ym)[:,-1:]
        y_pred = np.concatenate((y_pred, yp.detach().numpy()), axis = 1)
    return y_pred[:,1:]


# %% code to create forecast, err comp, plots of truth and prediction
# X_init = test_dataset[0][0].reshape(1, block_size, -1)
# print(X_init.shape)
n_steps = len(test_dataset.data) - 1
X_init = test_dataset.data[0].values.reshape(1, 1, -1)
y_pred = rolling_forecast(model_loaded, X_init, n_steps)
# %% 
# print(y_preds.shape)
#error calc

import matplotlib.pyplot as plt
y_test = test_dataset.data[1].values.reshape(1,1,-1)
for k in range(2,len(test_dataset.data)):
    y_test = np.concatenate([y_test, test_dataset.data[k].values.reshape(1,1,-1)],axis=1)
norm_vals = np.zeros(y_test.shape[1])
for i in range(len(norm_vals)):
    norm_vals[i] = np.linalg.norm(y_test.squeeze()[i].flatten())
max_norm = max(norm_vals)

# %%
figures_path = f"Figures/{model_path[7:]}/" 
err = np.zeros(y_pred.shape[1])

for i in range(len(err)):
    mse = (y_pred.squeeze()[i].flatten() - y_test.squeeze()[i].flatten())**2
    rel_mse = mse/max_norm**2
    err[i] = np.sum(rel_mse)

plt.plot(range(len(err)), err )
plt.xlabel("Time index (1-hour intervals)")
plt.ylabel(r"Forecast Relative MSE: $\sum (Y - y)^2 / ||Y||^2_2$")
plt.title(f"10m_u_component_of_wind")
plt.savefig(f"{figures_path}/testing_prediction_{model_path[7:]}")

# %% plots of y_preds and y_test:
import cartopy.crs as ccrs
lats = test_dataset.data['latitude'].values #121
lons = test_dataset.data['longitude'].values #240


# fig, axs = plt.subplots(1, 1, figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()})
# # Plot for truth values
# axs.coastlines()
# lons_2d, lats_2d = np.meshgrid(lons, lats)
# sc1 = axs.pcolormesh(lons_2d, lats_2d, y_test.squeeze()[0].reshape(48, 40).T, cmap='Spectral_r', vmin=0., vmax=1., transform=ccrs.PlateCarree())
# plt.colorbar(sc1, ax=axs, label='10m_u_component_of_wind')
# axs.set_title("Test value")
# plt.show()

for i in range(y_pred.shape[1]):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    # Plot for truth values
    axs[0].coastlines()
    lons_2d, lats_2d = np.meshgrid(lons, lats)
    sc1 = axs[0].pcolormesh(lons_2d, lats_2d, y_test.squeeze()[i].reshape(48, 40).T, cmap='Spectral_r', vmin=0., vmax=1., transform=ccrs.PlateCarree())
    plt.colorbar(sc1, ax=axs[0], label='10m_u_component_of_wind')
    axs[0].set_title("Test value")

    # Plot for prediction values
    axs[1].coastlines()
    sc2 = axs[1].pcolormesh(lons_2d, lats_2d, y_pred.squeeze()[i].reshape(48, 40).T, cmap='Spectral_r', vmin=0., vmax=1., transform=ccrs.PlateCarree())
    plt.colorbar(sc2, ax=axs[1], label='10m_u_component_of_wind')
    axs[1].set_title("Prediction value")
    
    plt.tight_layout()
    fig.suptitle(f"True value and Prediction for {test_dataset.data[i]['time'].values}, prediction = {i}")
    # plt.show()
    plt.savefig(f"{figures_path}/True value and Prediction for {test_dataset.data[i]['time'].values}_{model_path[7:]}.png")
    plt.close()


# y_pred = X_init
# n_steps = len(test_dataset.data) - 1
# ym = torch.tensor(y_pred[:,-block_size:])
# yp = model_loaded(ym)[:,-1:]
# y_pred = np.concatenate((y_pred, yp.detach().numpy()), axis = 1)
# y_pred2 = model_loaded(torch.tensor(y_pred))[:,-1:]
# print(y_pred2.shape)


# x,y = next(iter(test_loader))
# print(x.shape, y.shape)
# print(len(test_dataset))
# x,y =  test_dataset[0]
# print(x.shape, y.shape, type(x), type(y))
# test_inp = torch.randn((1,1,1920))
# test_out = model_loaded(test_inp)
# print(test_out.shape)

