# %% loading test data
from model import STN
from dataloader import load_data
import torch
from tqdm import tqdm

model = STN(sampling_size=(16,8))
obs_path = f"gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr"
data_sets, data_loaders = load_data(obs_path)

train_dataset = data_sets['train_dataset'] 
val_dataset = data_sets['val_dataset'] 
test_dataset = data_sets['test_dataset'] 

# %%
#need mean and stddev of training data
import numpy as np
train_data = train_dataset.data.to_numpy().flatten()
M = np.mean(train_data)
sdev = np.std(train_data)

# %% need X_test not X_test_norm
import xarray as xr
obs_path = f"gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr"
data = xr.open_zarr(obs_path)
data_u10 = data['10m_u_component_of_wind']
testing_period = ["2020-02-11T00:00:00.000000000","2020-02-20T23:00:00.000000000"]
X_test = data_u10.sel(time = slice(*testing_period)).to_numpy()

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
X_test_true = X_test
X_test = (X_test - M)/sdev
X_test_true = (X_test_true - M)/sdev
# %% Add noise to the truth to mimic observations
noise = 1
for k in range(1,np.size(X_test,0)):
    X_test[k-1,:]= X_test[k-1,:]+np.random.normal(0, noise, 2048)

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
        u = model(torch.tensor(np.reshape(ens[:,k],[1, 1, 64, 32]), dtype = torch.float32 ))
        u = u.detach().numpy()
        u_ensemble[:,k]=np.reshape(u,(64*32,))

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
model = STN((16,8))
model_path = "models/model_20240507_012704_4"

model.load_state_dict(torch.load(model_path))

# %%enkf
import scipy.io as sio
time = 40
n=int(64*32)
P=np.eye(n,n)

Q=0.03*np.eye(n,n)

R=0.0001

u_ensemble=np.zeros([64*32,2*64*32])
pred=np.zeros([time,64,32,1])

dt=10
count=0
for t in range(0, time, dt):
    
    for kk in range(0,dt-1):
        if (kk==0):   
            u=X_test[t+kk,:].reshape([1, 1, 64, 32])
            u=model(torch.tensor(u.reshape([1,1,64,32]), dtype=torch.float32)  )
            u = u.detach().numpy()
        else :
            u=model(torch.tensor(u, dtype = torch.float32))        
            u = u.detach().numpy() 
        pred[count,:,:,0]=np.reshape(u,[64,32])
        count=count+1
    x=u   
    x, P = ENKF(x, 2048, P, Q, R, X_test[t+dt,:], model,u_ensemble)

    print('output shape of ENKF', np.shape(x))
    
    pred[count,:,:,0]=np.reshape(x,[64,32])
    count=count+1

# sio.savemat('DA_every24HR_lead1200_everytime_noise_' + str(noise)+ '.mat',dict([('prediction',pred),('truth',np.reshape(X_test_true,[np.size(X_test,0),240,121,1])),('noisy_obs',np.reshape(X_test,[np.size(X_test,0),240,121,1]))]))
sio.savemat('DA_every10HR_lead3_everytime_noise_' + str(noise)+ '.mat',dict([('prediction',pred),('truth',np.reshape(X_test_true,[np.size(X_test,0),64,32,1])),('noisy_obs',np.reshape(X_test,[np.size(X_test,0), 64, 32,1]))]))

print('Done writing file')


# %%
