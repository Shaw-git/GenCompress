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
    

class JHTDBOverlap(BaseDataset):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = "JHTDB Latent"
        
        var = args["var"] if "var" in args else range(48)
        
        self.return_norm = args["return_norm"] if "return_norm" in args else False
        
        
        self.n_overlap = args["n_overlap"] if "n_overlap" in args else 0
        self.total_frame = args["total_frame"] if "total_frame" in args else 1
        
        
        self.data_input, self.bbp = self.load_jhtdb_dataset(self.data_path, var)
        self.data_input = self.data_input[:,:,:self.total_frame]
        
        self.shape = self.data_input.shape
        
        self.sp = [self.shape[0] ,self.shape[1], (self.shape[2]-self.n_frame)]
        
        # self.data_input, self.var_mean, self.var_scale = normalize_data(self.data_input, self.norm_type, axis=(1,2,3,4,5))
        
        
        print("Final Data Shape", self.data_input.shape)
        self.shape = self.data_input.shape
        self.delta_t = self.n_frame - self.n_overlap
        self.t_samples = (self.shape[2] - self.n_frame)//self.delta_t + 1
        
        assert((self.shape[2] - self.n_frame)%self.delta_t == 0)
        assert(self.inst_norm)
        
        
        # self.data_input = self.blocking_data(self.data_input)
        self.data_input = torch.FloatTensor(self.data_input)
        self.visble_length = self.update_length()
    
    def load_jhtdb_dataset(self, data_path, var):
        
        print("*************** Loading", self.dataset_name, "***************")
        data = np.load(data_path)
        
        var_data = data["data"][var]
        bbp = data["bbp"][var]
        
        self.pd_scale = data["scale"][var]
        self.pd_offset = data["offset"][var]
        
        print("Original Data Shape:", var_data.shape) #shape
        return var_data, bbp
    
    def update_length(self):
        self.dataset_length = self.shape[0] * self.shape[1] * self.t_samples
        return self.dataset_length           
        
        
    def __len__(self):
        return self.visble_length

    def __getitem__(self, idx):
        
        idx  = idx % self.dataset_length
        
        
        idx0 = idx//(self.shape[1]*self.t_samples)
        idx1 = idx//(self.t_samples)% self.shape[1]
        idx2 = idx%(self.t_samples)
        
        
        start_t , end_t = idx2*self.delta_t, idx2*self.delta_t+self.n_frame
        data = self.data_input[idx0, idx1, start_t:end_t]
        
        bbp = self.bbp[idx0, idx1, start_t:end_t]
        
        
    
        if self.inst_norm:
            data, offset, scale = self.apply_inst_norm(data, True)
        
        data = data.permute(1,0,2,3)
        
        return data, offset, scale, bbp, self.pd_offset[idx0, idx1, start_t:end_t], self.pd_scale[idx0, idx1, start_t:end_t]