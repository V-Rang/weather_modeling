# %% model initialization
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from BiLinearInterpolation import BilinearInterpolation
from utils import get_initial_weights

class STN(nn.Module):
    def __init__(self, sampling_size):
        super(STN,self).__init__()

        self.sampling_size = sampling_size

        self.pool = nn.MaxPool2d((2,2))

        self.conv1 = nn.Conv2d( 1, 32, (5,5), (1,1), 'same')
        self.conv2 = nn.Conv2d( 32, 32, (5,5), (1,1), 'same') 
        self.conv3 = nn.Conv2d( 1, 32, (2,2), (1,1), 'same') 
        self.conv4 = nn.Conv2d( 64, 32, (2,2), (1,1), 'same') 
        self.conv5 = nn.Conv2d( 32, 32, (2,2), (1,1), 'same') 
        self.conv6 = nn.Conv2d( 64, 32, (5,5), (1,1), 'same') 
        self.conv7 = nn.Conv2d( 32, 1, (5,5), (1,1), 'same')
        
        self.upsample = nn.Upsample(scale_factor=2)

        output_size = 50
        initial_weights, initial_biases = get_initial_weights(output_size)
        self.linear = nn.Linear(50, 6)
        self.linear.weight.data = initial_weights
        self.linear.bias.data = initial_biases
            
    def forward(self, x):
        self.input =  x
        conv1 = self.conv1(x)
        conv1 = F.relu(conv1)
        
        conv1 = self.conv2(conv1)
        conv1 = F.relu(conv1)   
        
        pool1 = self.pool(conv1)

        conv2 = self.conv2(pool1)
        conv2 = F.relu(conv2)
        
        conv2 = self.conv2(conv2)
        conv2 = F.relu(conv2)

        pool2 = self.pool(conv2)

        conv5 = self.conv2(pool2)
        conv5 = F.relu(conv5)

        conv5 = self.conv2(conv5)
        conv5 = F.relu(conv5)
        
        locnet = nn.Flatten()(conv5)
        locnet = nn.Linear(4096, 500)(locnet)
        locnet = F.relu(locnet)        
        locnet = nn.Linear(500, 200)(locnet)
        locnet = F.relu(locnet)
        locnet = nn.Linear(200, 100)(locnet)
        locnet = F.relu(locnet)
        locnet = nn.Linear(100, 50)(locnet)
        locnet = F.relu(locnet)
        locnet = self.linear(locnet)
        x = BilinearInterpolation(self.sampling_size)([self.input, locnet])
        up6 = torch.cat( ( F.relu(self.conv3(self.upsample(x))),conv2), dim = 1)
        conv6 = F.relu(self.conv4(up6))
        conv6 = F.relu(self.conv2(conv6))
        up7 = torch.cat( ( F.relu(self.conv5(self.upsample(conv6))),conv1), dim = 1)
        conv7 = F.relu(self.conv6(up7))         
        conv7 = F.relu(self.conv2(conv7))
        conv10 = self.conv7(conv7)        
        return conv10
