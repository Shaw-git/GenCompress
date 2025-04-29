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
    

class JHTDBLatent(BaseDataset):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = "JHTDB Latent"
        
        var = args["var"] if "var" in args else range(48)
        
        self.return_norm = args["return_norm"] if "return_norm" in args else False
        
        self.data_input = self.load_jhtdb_dataset(self.data_path, var)
        self.shape = self.data_input.shape
        
        self.sp = [self.shape[0] ,self.shape[1], (self.shape[2]-self.n_frame)]
        
        # self.data_input, self.var_mean, self.var_scale = normalize_data(self.data_input, self.norm_type, axis=(1,2,3,4,5))
        
        # self.data_input = self.blocking_data(self.data_input)
        self.data_input = torch.FloatTensor(self.data_input)
        
        self.visble_length = self.update_length()
    
    def load_jhtdb_dataset(self, data_path, var):
        
        print("*************** Loading", self.dataset_name, "***************")
        data = np.load(data_path)["data"][var]
        
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
                return data[0], data[1], data[2], 1
                
                
        # print(data.shape)
        return data