from torch.utils.data import Dataset
import numpy as np

def normalize_values(data):
    new_data = data.copy(deep=True)
    max_val = new_data.max()
    min_val = new_data.min()
    new_data = (new_data - min_val) / (max_val - min_val)
    return new_data

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

