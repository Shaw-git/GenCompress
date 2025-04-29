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
import h5py
from .normalization import normalize_data
from .basefunc import BaseDataset
from copy import deepcopy

class ERA5(BaseDataset):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = "ERA5"
        var  = args["var"]  if "var"  in args else ["temporature"]
        total_frame = args["total_frame"] if "total_frame" in args else 512
        

        self.data_input = self.load_era5_dataset(self.data_path, var)
        
        self.uniform_data_preprocessing()
        
        
        self.data_input, self.var_mean, self.var_scale = normalize_data(self.data_input, self.norm_type, axis=(0,1,2,3))
        
        self.data_input = self.blocking_data(self.data_input)
        
        self.data_input = torch.FloatTensor(self.data_input)
        
        self.visble_length = self.update_length()
    
    def load_era5_dataset(self, data_path, var):
        print("*************** Loading", self.dataset_name, "***************")
        all_data = np.load(data_path)["data"].astype(np.float32)
        print("Data Shape:", all_data.shape)
        return all_data

        
            
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