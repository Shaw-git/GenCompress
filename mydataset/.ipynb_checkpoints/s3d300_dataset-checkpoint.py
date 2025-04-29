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

class S3D300(BaseDataset):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = "S3D300"
        
        var  = args["var"]  if "var"  in args else np.arange(48)
        total_frame = args["total_frame"] if "total_frame" in args else [100,300]
        global_norm = args["global_norm"] if "global_norm" in args else True
        
        assert((total_frame[1]- total_frame[0])%self.n_frame==0)
        
        
        self.data_input = self.load_s3d_dataset(self.data_path, var, total_frame)
        self.uniform_data_preprocessing()
        
        if global_norm:
            self.data_input, self.var_mean, self.var_scale = normalize_data(self.data_input, self.norm_type, axis=(1,2,3)) #[60, 50, 512, 512]
        
        self.shape0 = self.data_input.shape
        
        self.data_input = self.blocking_data(self.data_input)
        self.shape1 = self.data_input.shape
        
        self.samples_each = np.prod(self.shape0[-3:])//np.prod(self.shape1[-3:])
        assert(np.prod(self.shape0[-3:])%np.prod(self.shape1[-3:])==0)
        
        print(self.samples_each,"samples for each species")
        
        self.data_input = torch.FloatTensor(self.data_input)
        self.visble_length = self.update_length()
        
        
    def load_s3d_dataset(self, data_path, var, total_frame):
        print("*************** Loading", self.dataset_name, "***************")
        file_name = ["input_0_1ms.npy", "input_1_2ms.npy", "input_2_3ms.npy"]
        intervals = [[0,100], [100,200], [200,300]]
        all_data = []
        for name, interval in zip(file_name, intervals):
            itv = [max(total_frame[0], interval[0])- interval[0], min(total_frame[1], interval[1]) - interval[0]]
            if itv[0]>=itv[1]:
                continue
            print("load",name, itv)
            data = np.load(data_path+"/"+name).reshape([-1, 640,640, 60]).transpose([3,0,1,2])[var, itv[0]:itv[1]]
            all_data.append(data)
        
        all_data = np.concatenate(all_data, axis = 1)[:,:,64:-64,64:-64]
            
        print("Original Data Shape:", all_data.shape)
        return all_data
        
        
    def update_length(self):
        self.dataset_length = self.data_input.shape[0]
        return self.dataset_length
    
    def __len__(self):
        return self.visble_length

    def __getitem__(self, idx):
        idx = idx % self.dataset_length
        
        species_idx = idx//self.samples_each
        
        data = self.data_input[idx]
        data = self.output_processing(data)
        
        a2,b2 = data["offset"], data["scale"]
        a1,b1 = self.var_mean.reshape(-1)[species_idx], self.var_scale.reshape(-1)[species_idx]
        data["offset"], data["scale"] = a1+b1*a2, b1*b2
        
        return data