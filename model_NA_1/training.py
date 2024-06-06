from model import STN
from dataloader import load_data
import torch
from tqdm import tqdm
import os
import numpy as np
model = STN(sampling_size=(12,10))
# obs_path = f"gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr"
obs_path = f"gs://weatherbench2/datasets/era5/1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr/"
_, data_loaders = load_data(obs_path)

train_loader = data_loaders['train_dataloader'] 
val_loader = data_loaders['val_dataloader'] 
test_loader = data_loaders['test_dataloader'] 
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#training one epoch:
def train_one_epoch():
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(tqdm(train_loader)):
        inputs, true_outputs = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, true_outputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:            
            last_loss = running_loss / 10 # loss per batch
            print('batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    return last_loss

from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
epoch_number = 0
EPOCHS = 100
best_vloss = 1_000_000.

average_batch_loss = []
validation_loss = []

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    model.train(True)
    avg_loss = train_one_epoch()
    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    average_batch_loss.append(avg_loss)
    validation_loss.append(avg_vloss)
    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'models/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)
    epoch_number += 1

figures_path = f"Figures/{model_path[7:]}/" 
if not os.path.exists(figures_path):
    os.makedirs(figures_path)

import matplotlib.pyplot as plt
xvals = np.arange(len(average_batch_loss))
# print(xvals, average_batch_loss, validation_loss)

plt.plot(xvals, average_batch_loss,label = 'training average batch loss')
plt.plot(xvals, validation_loss,label = 'validation loss')
plt.xlabel("Time index (1-hour intervals)")
plt.ylabel(r"Forecast Relative MSE: $\sum (Y - y)^2 / ||Y||^2_2$")
plt.title(f"Training and validation loss")
plt.legend()
plt.savefig(f"{figures_path}/training_validation_loss_{model_path[7:]}")
