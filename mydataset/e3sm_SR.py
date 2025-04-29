import torch
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import numpy as np
import json
from tqdm import tqdm
import threading


class ClimateInterp(Dataset):
    def __init__(self, data_path, recons_path, interp_path, mode = "combine"):
        
        self.dataset_name = "E3SM"
        # path = "/blue/ranka/xiao.li/dataset/climate_dataset/dataset_numpy/08.npz"
        self.e3sm_data = np.load(data_path)["data"][:,0]
        
        self.var_mean = np.mean(self.e3sm_data)
        self.var_std  =  np.std(self.e3sm_data)
        
        self.e3sm_recons = np.load(recons_path)["data"][:]
        self.e3sm_interp = np.load(interp_path)["data"][:]
        
        self.e3sm_data = (self.e3sm_data - self.var_mean)/self.var_std
        self.e3sm_recons = (self.e3sm_recons - self.var_mean)/self.var_std
        self.e3sm_interp = (self.e3sm_interp - self.var_mean)/self.var_std
        
        self.e3sm_data = torch.FloatTensor(self.e3sm_data).reshape([-1, 1, 240, 240])
        
        self.e3sm_recons = torch.FloatTensor(self.e3sm_recons).reshape([-1, 1, 240, 240])
        self.e3sm_interp = torch.FloatTensor(self.e3sm_interp).reshape([-1, 1, 240, 240])
        
        if mode == "combine":
            self.e3sm_input = torch.cat([self.e3sm_recons, self.e3sm_interp], dim=1)
            del self.e3sm_recons, self.e3sm_interp
            
        elif mode == "recons":
            self.e3sm_input = self.e3sm_recons
            del self.e3sm_interp
            
        elif mode == "interp":
            self.e3sm_input = self.e3sm_interp
            del self.e3sm_recons
        
        self.update_length()
            
    def update_length(self):
        self.dataset_length = self.e3sm_data.shape[0]
        return self.dataset_length            
        
        
    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        
        return self.e3sm_input[idx], self.e3sm_data[idx]

