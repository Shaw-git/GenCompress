import torch
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import numpy as np
import json
from tqdm import tqdm
import threading
import torch.nn.functional as F
import torchvision.transforms as T
from .normalization import normalize_data
from .basefunc import BaseDataset
from scipy.ndimage import zoom
import numpy as np

def downsampling_data(data, zoom_factors):
    # Apply zoom with specified factors for selective upsampling
    upsampled_data = zoom(data, zoom_factors, order=3)  # order=3 for cubic interpolation
    

class Climate240(BaseDataset):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = "E3SM"
        
        var = args["var"] if "var" in args else [0]
        
        self.downsampling  = args["downsampling"] if "downsampling" in args else 1
        
        
        self.data_input = self.load_e3sm_dataset(self.data_path, var)
        if self.downsampling >1:
            self.data_input = self.data_input[...,::self.downsampling,::self.downsampling]
            print("After downsampling:",self.data_input.shape)

        self.uniform_data_preprocessing()

        self.data_input, self.var_mean, self.var_scale = normalize_data(self.data_input, self.norm_type, axis=(1,2,3,4))
        
        self.data_input = self.blocking_data(self.data_input)
        self.data_input = torch.FloatTensor(self.data_input)
        
        self.visble_length = self.update_length()
    
    def load_e3sm_dataset(self, data_path, var):
        
        print("*************** Loading", self.dataset_name, "***************")
        data = np.load(data_path)["data"][:, np.asarray(var)] # [720, 5, 6, 240, 240]
        data = data.transpose([1,2,0,3,4])
        print("Original Data Shape:", data.shape) #shape (4, 6, 720, 240, 240)
        return data
    
    def update_length(self):
        self.dataset_length = self.data_input.shape[0]
        return self.dataset_length           
        
        
    def __len__(self):
        return self.visble_length

    def __getitem__(self, idx):
        idx = idx % self.dataset_length
        
        data = self.data_input[idx]
        data = self.output_processing(data)
        
        return data