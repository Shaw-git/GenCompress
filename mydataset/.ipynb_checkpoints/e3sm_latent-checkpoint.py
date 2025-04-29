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
    

class ClimateLatent(BaseDataset):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = "E3SM Latent"
        
        var = args["var"] if "var" in args else [0]
        self.seq = args["seq"] if "seq" in args else range(8)
        
        self.return_norm = args["return_norm"] if "return_norm" in args else False
        
        self.data_input = self.load_e3sm_dataset(self.data_path, var)
        self.shape = self.data_input.shape
        
        self.sp = [self.shape[0] ,self.shape[1], (self.shape[2]-self.n_frame)]
        
        self.data_input, self.var_mean, self.var_scale = normalize_data(self.data_input, self.norm_type, axis=(1,2,3,4,5))
        
        # self.data_input = self.blocking_data(self.data_input)
        self.data_input = torch.FloatTensor(self.data_input)
        
        self.visble_length = self.update_length()
    
    def load_e3sm_dataset(self, data_path, var):
        
        print("*************** Loading", self.dataset_name, "***************")
        latent_data = []
        
        for i in self.seq:
            data = np.load(data_path+"/latent_data_%02d.npz"%(i))["data"] # (4, 6, 720, 240, 240)
            latent_data.append(data)
        data = np.concatenate(latent_data, axis = 2)
        
        print("Original Data Shape:", data.shape) #shape
        return data
    
    def update_length(self):
        self.dataset_length = self.shape[0] * self.shape[1] * (self.shape[2]-self.n_frame)
        return self.dataset_length           
        
        
    def __len__(self):
        return self.visble_length

    def __getitem__(self, idx):
        idx0 = idx//(self.sp[1]*self.sp[2])
        idx1 = idx//(self.sp[2])% self.sp[1]
        idx2 = idx%(self.sp[2])
        
        # print(idx0, idx1, idx2)
        data = self.data_input[idx0, idx1, idx2:idx2+self.n_frame]
        # print(data.shape)
        data = torch.transpose(data, 1,0)
        
        if self.inst_norm:
            data = self.apply_inst_norm(data, self.return_norm)
            if self.return_norm:
                a2,b2 = data[1], data[2]
                a1,b1 = self.var_mean.reshape(-1), self.var_scale.reshape(-1)
                
                offset, scale = a1+b1*a2, b1*b2
                return data[0], offset, scale
                
                
        # print(data.shape)
        return data