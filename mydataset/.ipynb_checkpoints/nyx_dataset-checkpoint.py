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


class NYX(BaseDataset):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = "NYX"
        var  = args["var"]  if "var"  in args else ["velocity_x"]
        total_frame = args["total_frame"] if "total_frame" in args else 512
        

        self.data_input = self.load_nyx_dataset(self.data_path, var, total_frame)
        
        self.uniform_data_preprocessing()
        
        
        self.data_input, self.var_mean, self.var_scale = normalize_data(self.data_input, self.norm_type, axis=(1,2,3))
        
        self.data_input = self.blocking_data(self.data_input)
        
        self.data_input = torch.FloatTensor(self.data_input)
        
        self.visble_length = self.update_length()
    
    def load_nyx_dataset(self, data_path, var, total_frame):
        print("*************** Loading", self.dataset_name, "***************")
        print("Loading:", var)
        with h5py.File(data_path, 'r') as hdf:
            # List all groups
            hdf  = hdf['native_fields']
            all_data = []
            for v in var:
                all_data.append(hdf[v][0:total_frame].squeeze())

        all_data = np.stack(all_data)
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