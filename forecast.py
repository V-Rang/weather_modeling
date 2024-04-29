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

#implementing rolling forecast
# (0...5) -> (6...11) | compare to (6...11) of actual
# (6...11) -> (12...17) | compare to (12...17) of actual
# (12...17) -> (18...23) | compare to (18...23) of actual
# (18...23) -> (24...29) | compare to (24...29) of actual
# (24...29) -> (30...35) | compare to (30...35) of actual
def rolling_forecast(model, Xt, n_steps):
    # Initialize states
    y_pred = Xt
    in_size = y_pred.shape[0]

    # Iterate over timesteps
    for k in range(n_steps):
        ym = torch.tensor(y_pred[-in_size:].reshape(1,*y_pred[-in_size:].shape))
        yp = model(ym)
        y_pred = np.concatenate((y_pred, yp.detach().numpy().squeeze()), axis=0)        
    return y_pred[Xt.shape[0] :]

#predictions
n_steps = (test_dataset.data.shape[0] - in_len + 1) // out_len 
X_init = test_dataset.data[: in_len].to_numpy().astype(np.float32)
y_pred = X_init

in_size =  X_init.shape[0]

y_pred = rolling_forecast(
    model_loaded,
    X_init,
    n_steps
)

## error computation between prediction and true values
y_test = test_dataset.data[in_len:in_len+out_len].values

for i in range(1,n_steps):
    y_test = np.concatenate( (y_test, test_dataset.data[in_len*(i+1):in_len*(i+1)+out_len].values),axis = 0)

norm_vals = np.zeros(len(y_test))

for i in range(len(y_test)):
    norm_vals[i] = np.linalg.norm(y_test[i])

max_norm = max(norm_vals)

err = np.zeros(y_pred.shape[0])
for i in range(y_pred.shape[0]):
    mse = (y_pred[i] - y_test[i])**2
    rel_mse = mse/max_norm**2
    err[i] = np.sum(rel_mse)
plt.plot(range(len(err)), err )
plt.xlabel("Time index (6-hour intervals)")
plt.ylabel(r"Forecast Relative MSE: $\sum (Y - y)^2 / ||Y||^2_2$")
plt.title(f"10m_u_component_of_wind")
plt.savefig(f"{figures_path}/testing_prediction_{model_path[7:]}")

# # plots
lats = test_dataset.data['latitude'].values #121
lons = test_dataset.data['longitude'].values #240


#series of images:
for i in range(0,30):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    # Plot for truth values
    axs[0].coastlines()
    lons_2d, lats_2d = np.meshgrid(lons, lats)
    sc1 = axs[0].pcolormesh(lons_2d, lats_2d, y_test[i].T, cmap='Spectral_r', vmin=0., vmax=1., transform=ccrs.PlateCarree())
    plt.colorbar(sc1, ax=axs[0], label='10m_u_component_of_wind')
    axs[0].set_title("Test value")

    # Plot for prediction values
    axs[1].coastlines()
    sc2 = axs[1].pcolormesh(lons_2d, lats_2d, y_pred[i].T, cmap='Spectral_r', vmin=0., vmax=1., transform=ccrs.PlateCarree())
    plt.colorbar(sc2, ax=axs[1], label='10m_u_component_of_wind')
    axs[1].set_title("Prediction value")
    plt.tight_layout()
    fig.suptitle(f"True value and Prediction for {test_dataset.data[i+6]['time'].values}, prediction = {i}")
    # plt.show()
    plt.savefig(f"{figures_path}/True value and Prediction for {test_dataset.data[i+6]['time'].values}_{model_path[7:]}.png")
    plt.close()