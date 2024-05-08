# %% prelim stuff
from model import SegNet
from dataloader import load_data
import torch
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import os

model_loaded = SegNet(in_len = 6, out_len = 6)
model_path = "models/model_20240429_130735_73"
figures_path = f"Figures/{model_path[7:]}/" 

if not os.path.exists(figures_path):
    os.makedirs(figures_path)


model_loaded.load_state_dict(torch.load(model_path))

obs_path = f"gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
datasets, _ = load_data(obs_path)
train_dataset = datasets['train_dataset']
val_dataset = datasets['val_dataset']
test_dataset = datasets['test_dataset']

in_len, out_len = 6, 6

# %% if  1-1 reconstr is possible
# (0...5) -> (6...11) | compare to (6...11) of actual

# y_test = test_dataset.data[0:6]
# print(y_test.shape)

# y_pred = model_loaded(y_test)
# X_init = test_dataset.data[: in_len].to_numpy().astype(np.float32)

# input = torch.tensor(X_init[:].reshape(1,*X_init[:].shape))
# # print(type(input),input.shape)
# output = model_loaded(input)

# test =  output[:,0,:].detach().numpy()
# test = test.reshape(test.shape[1],test.shape[2])

input = test_dataset.data[:in_len].to_numpy().astype(np.float32)
# print(input.shape)
# input_plotter = input[0]
# print(input_plotter.shape)
input_model = torch.tensor(input[:].reshape(1,*input[:].shape))
output_model = model_loaded(input_model)

output =  output_model[:].detach().numpy()
output = output.reshape(output.shape[1:])
print(input.shape, output.shape)


test_prediction = test_dataset.data[in_len:in_len+out_len].astype(np.float32)

# %% plotter
lats = test_dataset.data['latitude'].values #121
lons = test_dataset.data['longitude'].values #240

fig, axs = plt.subplots(1, 2, figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()})
i = 0
# Plot for truth values
axs[0].coastlines()
lons_2d, lats_2d = np.meshgrid(lons, lats)
sc1 = axs[0].pcolormesh(lons_2d, lats_2d, test_prediction[i].T, cmap='Spectral_r', vmin=0., vmax=1., transform=ccrs.PlateCarree())
plt.colorbar(sc1, ax=axs[0], label='10m_u_component_of_wind')
axs[0].set_title("Test value")

# Plot for prediction values
axs[1].coastlines()
sc2 = axs[1].pcolormesh(lons_2d, lats_2d, output[i].T, cmap='Spectral_r', vmin=0., vmax=1., transform=ccrs.PlateCarree())
plt.colorbar(sc2, ax=axs[1], label='10m_u_component_of_wind')
axs[1].set_title("Prediction value")
plt.tight_layout()
fig.suptitle(f"True value and Prediction for {test_dataset.data[i+6]['time'].values}, prediction = {i}")
# plt.show()
plt.savefig(f"{figures_path}/True value and Prediction for {test_dataset.data[i+6]['time'].values}_{model_path[7:]}.png")
plt.show()
# plt.close()





