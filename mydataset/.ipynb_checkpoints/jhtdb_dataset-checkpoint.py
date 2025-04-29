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

class JHTDB(BaseDataset):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = "JHTDB"
        var  = args["var"]  if "var"  in args else ["isotropic"]
        total_frame = args["total_frame"] if "total_frame" in args else 512
        

        self.data_input = self.load_jhtdb_dataset(self.data_path, var, total_frame)
        self.uniform_data_preprocessing()
        
        if not self.inst_norm:
            print(self.data_input.shape)
            self.data_input, self.var_mean, self.var_scale = normalize_data(self.data_input, self.norm_type, axis=(0,1,2,3))
        
        self.data_input = self.blocking_data(self.data_input)
        self.data_input = torch.FloatTensor(self.data_input)
        
        self.visble_length = self.update_length()
    
    def load_jhtdb_dataset(self, data_path, var, total_frame):
        print("*************** Loading", self.dataset_name, "***************")
        print("Loading:", var)
        
        name_map = {"PSL":"Pressure_0001","Velocity":"Velocity_0001"}
        
        all_data = []

        if "PSL" in var:
            cur_path = data_path+"/channel_psl.h5"
            with h5py.File(cur_path, 'r') as hdf:
                all_data.append(hdf[name_map["PSL"]][0:total_frame].squeeze())
     
                
        if "Velocity" in var:
            cur_path = data_path+"/channel.h5"
            with h5py.File(cur_path, 'r') as hdf:
                all_data.append(hdf[name_map["Velocity"]][0:total_frame].squeeze()[...,0])

                
        if 'Time' in var:
            cur_path = data_path+"/channel_txy_z512.h5"
            with h5py.File(cur_path, 'r') as hdf:
                data = [hdf["Pressure_%04d"%i][:] for i in range(1,513)] # Replace with the dataset name from the keys
                data = np.stack(data).squeeze()
                
                all_data.append(data)
                
        if "isotropic_48" in var:
            cur_path = data_path+'/JHTDB_isotropic_48.npz'  # Replace with the actual file path
            data = np.load(cur_path)["data"].transpose([1,0,2,3])  # [64, 256, 512, 512]
            data = data[:,0:total_frame]
            all_data.append(data)
        
        if "isotropic_16" in var:
            cur_path = data_path+'/JHTDB_isotropic_16.npz'  # Replace with the actual file path
            data = np.load(cur_path)["data"].transpose([1,0,2,3])  # [16, 256, 512, 512]
            data = data[:,0:total_frame]
            all_data.append(data)
            
                

        all_data = np.concatenate(all_data)
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
        
        if not self.inst_norm:
            data["scale"] = self.var_scale.reshape([-1])
            data["offset"] = self.var_mean.reshape([-1])
            
        
        return data