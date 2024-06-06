from torch.utils.data import Dataset
import numpy as np
import torch

def normalize_values(data): 
    new_data = data.copy(deep=True)
    max_val = new_data.max()
    min_val = new_data.min()
    new_data = (new_data - min_val) / (max_val - min_val)
    return new_data

class Generate_Data(Dataset):
    def __init__(self,data, block_length = 8):
        self.data = data
        self.block_length = block_length
        self.data_len = len(self.data) 
    
    def __len__(self):
        return max(0, self.data_len - self.block_length)
    
    def __getitem__(self,idx):
        in_start = idx
        in_end = in_start + self.block_length
        out_start = in_start + 1
        out_end = out_start + self.block_length
        return (self.data[in_start:in_end].to_numpy().astype(np.float32), self.data[out_start:out_end].to_numpy().astype(np.float32))
