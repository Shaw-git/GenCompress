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

class Video(BaseDataset):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = "Video"
        
        var  = args["var"]  if "var"  in args else np.arange(10)
        
        assert(8%self.n_frame==0)
        
        
        self.data_input = self.load_video_dataset(self.data_path, var)
        self.uniform_data_preprocessing()
        
        # self.data_input, self.var_mean, self.var_scale = normalize_data(self.data_input, self.norm_type, axis=(0,1,2,3))
        self.data_input = self.blocking_data(self.data_input)
        self.data_input = torch.FloatTensor(self.data_input)
        
    
        self.visble_length = self.update_length()
        
    def load_video_dataset(self, data_path, var):
        print("*************** Loading", self.dataset_name, "***************")
        all_data = []
        for cur_v in var:
            cur_path = data_path+"/%03d.npz"%(cur_v)
            data = np.load(cur_path)["data"].astype(np.float32)[:,:,::2,::2]# [1000, 8, 640, 640]
            data, self.var_mean, self.var_scale = normalize_data(data, self.norm_type, axis=(0,1,2,3))
            all_data.append(data)
            
        all_data = np.concatenate(all_data, axis = 0)
        
        print("Original Data Shape:", all_data.shape)
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
        data["input"] = data["input"][None].expand(3,-1,-1,-1)
        return data