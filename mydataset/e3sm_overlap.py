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
    

class ClimateOverlap(BaseDataset):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = "E3SM"
        
        var = args["var"] if "var" in args else [0]
        self.n_overlap = args["n_overlap"] if "n_overlap" in args else 0
        
        self.extra_n_frame = args["extra_n_frame"] if "extra_n_frame" in args else 1
        self.extra_path = args["extra_path"]
        
        self.downsampling  = args["downsampling"] if "downsampling" in args else 1
        
        self.data_input = self.load_e3sm_dataset(self.data_path, var)
        self.extra_frame = self.load_e3sm_dataset(self.extra_path, var)[:,:,0:self.extra_n_frame,:,:]
        self.data_input = np.concatenate([self.data_input , self.extra_frame], axis = 2)
    
    
    
        self.shape = self.data_input.shape
        self.delta_t = self.n_frame - self.n_overlap
        self.t_samples = (self.shape[2] - self.n_frame)//self.delta_t + 1
        
        assert((self.shape[2] - self.n_frame)%self.delta_t == 0)
        assert(self.inst_norm)
        
        
        if self.downsampling >1:
            self.data_input = self.data_input[...,::self.downsampling,::self.downsampling]
            print("After downsampling:",self.data_input.shape)

        # self.uniform_data_preprocessing()

        # self.data_input, self.var_mean, self.var_scale = normalize_data(self.data_input, self.norm_type, axis=(1,2,3,4))
        # self.data_input = self.blocking_data(self.data_input)
        self.data_input = torch.FloatTensor(self.data_input)
        
        self.visble_length = self.update_length()
    
    def load_e3sm_dataset(self, data_path, var):
        
        print("*************** Loading", self.dataset_name, "***************")
        data = np.load(data_path)["data"][:, np.asarray(var)] # [720, 5, 6, 240, 240]
        data = data.transpose([1,2,0,3,4])
        print("Original Data Shape:", data.shape) #shape (4, 6, 720, 240, 240)
        return data
    
    def update_length(self):
        self.dataset_length = self.shape[0] * self.shape[1] * self.t_samples
        return self.dataset_length           
        
        
    def __len__(self):
        return self.visble_length

    def __getitem__(self, idx):
        
        idx = idx % self.dataset_length
        
        
        idx0 = idx//(self.shape[1]*self.t_samples)
        idx1 = idx//(self.t_samples)% self.shape[1]
        idx2 = idx%(self.t_samples)
        
        
        start_t , end_t = idx2*self.delta_t, idx2*self.delta_t+self.n_frame
        data = self.data_input[idx0, idx1, start_t:end_t]
        # print(idx0, idx1, "t",start_t , end_t)
    
        data = self.output_processing(data)
        
        # offset, scale = data["offset"], data["scale"]
        
        
        return data