from model import SegNet
from dataloader import load_data
import torch
from tqdm import tqdm

model = SegNet(in_len=6, out_len=6)
obs_path = f"gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
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
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss


from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

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

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'models/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

