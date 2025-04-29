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
        
        # self.downsampling  = args["downsampling"] if "downsampling" in args else 1
        
        self.data_input, self.bbp = self.load_e3sm_dataset(self.data_path, var)
        self.extra_frame, self.bbp_extra = self.load_e3sm_dataset(self.extra_path, var)
        
        self.extra_frame = self.extra_frame[:,:,0:self.extra_n_frame,:,:]
        self.bbp_extra = self.bbp_extra[:,:,0:self.extra_n_frame]
        
        self.data_input = np.concatenate([self.data_input , self.extra_frame], axis = 2)
        self.bbp = np.concatenate([self.bbp , self.bbp_extra], axis = 2)
    
    
        print("Final Data Shape", self.data_input.shape)
        self.shape = self.data_input.shape
        self.delta_t = self.n_frame - self.n_overlap
        self.t_samples = (self.shape[2] - self.n_frame)//self.delta_t + 1
        
        assert((self.shape[2] - self.n_frame)%self.delta_t == 0)
        assert(self.inst_norm)
        
        
        # if self.downsampling >1:
        #     self.data_input = self.data_input[...,::self.downsampling,::self.downsampling]
        #     print("After downsampling:",self.data_input.shape)

        # self.uniform_data_preprocessing()

        # self.data_input, self.var_mean, self.var_scale = normalize_data(self.data_input, self.norm_type, axis=(1,2,3,4))
        # self.data_input = self.blocking_data(self.data_input)
        self.data_input = torch.FloatTensor(self.data_input)
        
        self.visble_length = self.update_length()
    
    def load_e3sm_dataset(self, data_path, var):
        
        print("*************** Loading", self.dataset_name, "***************")
        npz_data  =np.load(data_path)
        data = npz_data["data"][np.asarray(var)] # [1, 6, 720, 64, 16, 16]
        bbp = npz_data["bbp"].reshape([-1, 6, 720])[np.asarray(var)]
        
        print("Original Data Shape:", data.shape, "BBP", bbp.shape) #shape (4, 6, 720, 240, 240)
        return data, bbp
    
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
        
        bbp = self.bbp[idx0, idx1, start_t:end_t]
    
        if self.inst_norm:
            data, offset, scale = self.apply_inst_norm(data, True)
        
        data = data.permute(1,0,2,3)
        return data, offset, scale, bbp