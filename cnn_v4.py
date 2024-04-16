# Training data: Jan 1 - Jan 31, 2022
# Testing data: Feb 1 - Feb 10, 2022
# Input, output length = 6 each
# So, the network I/O looks like
# [0,..,5] -> [6,..,11]
# [1,..,6] -> [7,..,12]
# ....
# [228,...,233] -> [234,...,239]
# relevant results of model in Experiment 1: https://www.overleaf.com/read/qjgqjmbdbwbw#3b14be



from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader
import torchvision
import torch

# data = xr.open_dataset("../ds1.nc") #8760 time values

data_path = "/scratch1/09052/venurang/ds1.nc"
data = xr.open_dataset(data_path) #8760 time values

train_period = ["2022-01-01T00:00:00.000000000","2022-01-31T23:00:00.000000000"]
test_period = ["2022-02-01T00:00:00.000000000","2022-02-10T23:00:00.000000000"]

X_train = data['u10'].sel(time = slice(*train_period)) #xarray.core.dataarray.DataArray
X_test = data['u10'].sel(time = slice(*test_period)) #xarray.core.dataarray.DataArray

# print(len(X_train), len(X_test)) #744, 240

#training
cur_min, cur_max = float('inf'), float('-inf')
for i in range(len(X_train)):
    data_min, data_max = X_train[i].to_numpy().min(), X_train[i].to_numpy().max()
    cur_min = min(cur_min, data_min)
    cur_max = max(cur_max, data_max) 

for i in range(len(X_train)):
    X_train[i].to_numpy()[:] =  (X_train[i].to_numpy()[:] - cur_min)/(cur_max - cur_min)
    

#Testing:
cur_min, cur_max = float('inf'), float('-inf')
for i in range(len(X_test)):
    data_min, data_max = X_test[i].to_numpy().min(), X_test[i].to_numpy().max()
    cur_min = min(cur_min, data_min)
    cur_max = max(cur_max, data_max) 

for i in range(len(X_test)):
    X_test[i].to_numpy()[:] =  (X_test[i].to_numpy()[:] - cur_min)/(cur_max - cur_min)


class Generate_Data(Dataset):
    def __init__(self,data, in_len=1,out_len=1):
        self.data = data
        self.in_len = in_len
        self.out_len = out_len
        self.max_start_index = len(self.data) - self.in_len - self.out_len
    
    def __len__(self):
        return max(0, self.max_start_index + 1)
    
    def __getitem__(self,idx):
        in_start = idx
        in_end = in_start + self.in_len

        out_start = in_end
        out_end = out_start + self.out_len

        return (self.data[in_start:in_end].to_numpy().astype(np.float32), self.data[out_start:out_end].to_numpy().astype(np.float32))


in_len = 6
out_len = 6

train_dataset = Generate_Data(X_train, in_len, out_len)
train_loader = DataLoader(train_dataset)

test_dataset = Generate_Data(X_test, in_len, out_len)
test_loader = DataLoader(test_dataset)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self,in_size,out_size):
        super().__init__()
        self.in_size = in_size #number of slices as input
        self.out_size =  out_size #number of slices to forecast
        self.conv1 = nn.Conv2d(self.in_size, 10, 3, padding = 1)
        self.conv2 = nn.Conv2d(10, 20, 3, padding = 1)
        self.conv3 = nn.Conv2d(20, self.out_size, 3, padding = 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

net = Net(in_len,out_len)


criterion = nn.MSELoss() #have to experiment with different losses
optimizer = optim.Adam(net.parameters(),lr=1e-3) 

training_losses = np.zeros(len(train_dataset))

for i in range(len(train_dataset)):
    x,y = next(iter(train_loader))
    optimizer.zero_grad()
    outputs = net(x)
    loss = criterion(outputs,y)
    loss.backward()
    optimizer.step()
    training_losses[i] = loss.item()

torch.save(net.state_dict(),"../models/test_model_monthly_2.pth")

np.save("train_year_results_month_Jan_2022_2.npy",training_losses)

#loading model
net_test = Net(in_len,out_len)
net_test.load_state_dict(torch.load("../models/test_model_monthly_2.pth"))
criterion = nn.MSELoss() #have to experiment with different losses
optimizer = optim.Adam(net_test.parameters(),lr=1e-3) 

testing_losses  = np.zeros(len(test_dataset))

for i in range(len(testing_losses)):
    x,y = next(iter(test_loader))
    outputs = net_test(x)
    testing_losses[i] = criterion(outputs,y).item()


np.save("test_February_results_month_Jan_2022_2.npy",testing_losses)
