import scipy.io
import torch
model_path = "models/model_20240531_222522_99"
mat = scipy.io.loadmat("DA_every10HR_lead3_everytime_noise_model_20240531_222522_99_0.001.mat")
prediction = mat['prediction']
truth = mat['truth']
noisy_obs = mat['noisy_obs']
loss_fn = torch.nn.MSELoss()
pred_truth_diff = []
pred_obs_diff = []

for i in range(prediction.shape[0]):
    pred_truth_diff.append(loss_fn(torch.from_numpy(prediction[i]), torch.from_numpy(truth[i]) ))
    pred_obs_diff.append(loss_fn(torch.from_numpy(prediction[i]), torch.from_numpy(noisy_obs[i]) ))
    
# %% making plot
import os
figures_path = f"Figures/{model_path[7:]}/" 
if not os.path.exists(figures_path):
    os.makedirs(figures_path)
    
    
import os
figures_path = f"Figures/{model_path[7:]}/" 
if not os.path.exists(figures_path):
    os.makedirs(figures_path)
    
import matplotlib.pyplot as plt
import numpy as np
xvals = np.arange(prediction.shape[0])
plt.plot(xvals, pred_truth_diff, label = 'Prediction v/s Truth')
plt.plot(xvals, pred_obs_diff, label = 'Prediction v/s Noise added observations')
plt.xlabel("Time index (1-hour intervals)")
plt.ylabel(r"Forecast Relative MSE: $\sum (Y - y)^2 / ||Y||^2_2$")
plt.title(f"Testing Loss")
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.15), ncol=2)
# plt.legend(loc = 'upper right') 
plt.savefig(f"{figures_path}/testing_loss_{model_path[7:]}")


# import matplotlib.pyplot as plt
# import numpy as np
# xvals = np.arange(prediction.shape[0])
# plt.plot(xvals, calc_losses,label = 'Testing Loss')
# plt.xlabel("Time index (1-hour intervals)")
# plt.ylabel(r"Forecast Relative MSE: $\sum (Y - y)^2 / ||Y||^2_2$")
# plt.title(f"Testing Loss")
# plt.legend() 
# plt.savefig(f"{figures_path}/testing_loss_{model_path[7:]}")