from torch.utils.data import Dataset
import numpy as np

class CoresetWrapper(Dataset):
    def __init__(self, ori_dataset):
        self.ori_dataset = ori_dataset
        self.selected_indices = np.array([0])
    
    def __len__(self):
        return len(self.selected_indices)
    
    def __getitem__(self, idx):
        return self.ori_dataset[self.selected_indices[idx]]
    
    def set_indices(self, selected_indices):
        self.selected_indices = selected_indices