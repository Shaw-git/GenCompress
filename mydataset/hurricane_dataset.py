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
import gzip

def read_gzip_data(filename, xdim=500, ydim=500, zdim=100):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype='>f4')  # '>f4' is big-endian float32
    return data.reshape((zdim, ydim, xdim))

def load_hurricane_dataset(path, variables, z_step=2, remove_gnd = True):
    
    map_name = {"temperature": "TC", "pressure": "P", "windx":"U", "windy":"V","windz":"w"}
    all_data = []
    
    for var in variables:
        var_data = []
        for i in range(1,48):
            cur_p = path + "/%sf%02d.bin.gz"%(map_name[var],i)
            data = read_gzip_data(cur_p)
            if remove_gnd:
                data = data[36:]
            if z_step!=1:
                data = data[::z_step]
            var_data.append(data)
        var_data = np.stack(var_data)
        all_data.append(var_data)
    all_data = np.stack(all_data)
    return all_data


class Hurricane(BaseDataset):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = "Hurricane"
        var  = args["var"]  if "var"  in args else ["temperature","pressure"]
        
        print("*************** Loading", self.dataset_name, "***************")
        
        self.data_input = load_hurricane_dataset(self.data_path, var)
        
        self.uniform_data_preprocessing()
        
        self.data_input, self.var_mean, self.var_scale = normalize_data(self.data_input, self.norm_type, axis=(1,2,3,4))
        
        self.data_input = self.blocking_data(self.data_input)
        self.data_input = torch.FloatTensor(self.data_input)
        
        self.visble_length = self.update_length()
      
            
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