# %% loading test data
from model import STN
from dataloader import load_data
import torch
from tqdm import tqdm
import scipy.io as sio
import numpy as np
from utils import normalize_values

model = STN(sampling_size=(12,10))
obs_path = f"gs://weatherbench2/datasets/era5/1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr/"
data_sets, data_loaders = load_data(obs_path)
train_dataset = data_sets['train_dataset'] 
val_dataset = data_sets['val_dataset'] 
test_dataset = data_sets['test_dataset'] 

# %%
import numpy as np

# %% need X_test not X_test_norm
import xarray as xr
import copy
# obs_path = f"gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr"
obs_path = f"gs://weatherbench2/datasets/era5/1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr/"
data = xr.open_zarr(obs_path)
data_u10 = data['10m_u_component_of_wind']
testing_period = ["2020-02-11T00:00:00.000000000","2020-02-20T23:00:00.000000000"]
longitude_range = [229.49999999999997, 299.99999999999994]
latitude_range = [11.999999999999991, 70.49999999999999]
X_test = data_u10.sel(time = slice(*testing_period))
X_test = X_test.sel(latitude = slice(*latitude_range))
X_test = X_test.sel(longitude = slice(*longitude_range))
lats = X_test['latitude']
lons = X_test['longitude']
num_x, num_y = len(lons), len(lats) #48, 40
print(num_x, num_y)
    
X_test = normalize_values(X_test).to_numpy()
X_test_true = copy.deepcopy(X_test)

X_test_obs = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

# print(X_test_obs.shape)
# X_test = X_test
# X_test_true = X_test_true
# %% Add noise to the truth to mimic observations
noise = 1e-3
for k in range(1,np.size(X_test_obs,0)):
    # X_test_obs[k-1,:]= X_test_obs[k-1,:]+np.random.normal(0, noise, 2048)
    X_test_obs[k-1,:]= X_test_obs[k-1,:] + np.random.normal(0, noise, num_x * num_y)

def ENKF(x, n, P ,Q, R, obs, model, u_ensemble):
    obs=np.reshape(obs,[n,1]) 
    x=np.reshape(x,[n,1])
    [U,S,V]=np.linalg.svd(P)
    D=np.zeros([n,n])
    np.fill_diagonal(D,S)
    sqrtP=np.dot(np.dot(U,np.sqrt(D)),U)
    ens=np.zeros([n,2*n])
    ens[:,0:n]=np.tile(x,(1,n)) + sqrtP
    ens[:,n:]=np.tile(x,(1,n)) - sqrtP
    ## forecasting step,dummy model

    for k in range(0, np.size(ens,1)):
        # u = model(torch.tensor(np.reshape(ens[:,k],[1, 1, 64, 32]), dtype = torch.float32 ))
        u = model(torch.tensor(np.reshape(ens[:,k],[1, 1, num_x, num_y]), dtype = torch.float32 ))
        u = u.detach().numpy()
        u_ensemble[:,k]=np.reshape(u,(num_x*num_y,))

    ############################
    x_prior = np.reshape(np.mean(u_ensemble,1),[n,1])
    print('shape pf x_prior',np.shape(x_prior))
    print('shape pf obs',np.shape(obs))
    cf_ens = ens - np.tile(x_prior,(1,2*n))
    P_prior = np.dot(cf_ens,np.transpose(cf_ens))/(2*n - 1)+Q
    h_ens = ens
    y_prior=np.reshape(np.mean(h_ens,1),[n,1])
    ch_ens = h_ens - np.tile(y_prior,(1,2*n))
    print('shape pf y_prior',np.shape(y_prior))
    P_y = np.dot(ch_ens, np.transpose(ch_ens))/(2*n-1) + R
    P_xy = np.dot(cf_ens, np.transpose(ch_ens)) /(2*n-1)
    K = np.dot(P_xy,np.linalg.inv(P_y))
    P = P_prior - np.dot(np.dot(K,P_y),np.transpose(K))
    x = x_prior + np.dot(K,(obs-y_prior))

    return x, P


# %%

model = STN((12,10))
# model_path = "models/model_20240507_013640_99"
model_path = "models/model_20240531_222522_99"

model.load_state_dict(torch.load(model_path))

# %%enkf
import scipy.io as sio
# time = 40
time = X_test.shape[0]
n=int(num_x*num_y)
P=np.eye(n,n)
Q=0.03*np.eye(n,n)
R=0.0001

# u_ensemble=np.zeros([64*32,2*64*32])
# pred=np.zeros([time,64,32,1])

u_ensemble = np.zeros([num_x*num_y,2*num_x*num_y])
pred = np.zeros([time,num_x,num_y,1])

dt = 10
count=0
for t in range(0, time, dt):
    for kk in range(0,dt-1):
        if (kk==0):   
            # u=X_test_obs[t+kk,:].reshape([1, 1, 64, 32])
            # u=model(torch.tensor(u.reshape([1,1,64,32]), dtype=torch.float32)  
            u=X_test_obs[t+kk,:].reshape([1, 1, num_x, num_y])
            u=model(torch.tensor(u.reshape([1,1, num_x, num_y]), dtype=torch.float32)  )
            u = u.detach().numpy()
        else :
            u=model(torch.tensor(u, dtype = torch.float32))        
            u = u.detach().numpy() 
        # pred[count,:,:,0]=np.reshape(u,[64,32])
        pred[count,:,:,0]=np.reshape(u,[num_x, num_y])        
        count=count+1
    x = u   
    # x, P = ENKF(x, 2048, P, Q, R, X_test_obs[t+dt-1,:], model,u_ensemble)
    x, P = ENKF(x, num_x * num_y, P, Q, R, X_test_obs[t+dt-1,:], model,u_ensemble)

    print('output shape of ENKF', np.shape(x))
    
    # pred[count,:,:,0]=np.reshape(x,[64,32])
    pred[count,:,:,0]=np.reshape(x,[num_x, num_y])
    count=count+1

sio.savemat('DA_every10HR_lead3_everytime_noise_' + str(model_path[7:]) + '_' + str(noise)+ '.mat',dict([('prediction',pred),('truth',np.reshape(X_test_true,[np.size(X_test_true,0),num_x,num_y,1])),('noisy_obs',np.reshape(X_test_obs,[np.size(X_test_true,0), num_x, num_y,1]))]))


