import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as T
from copy import deepcopy
import numpy as np

import torch
import torch.nn.functional as F
import math


def block_hw(data, block_size=(256, 256)):
    V, S, T, H, W = data.shape
    h_block, w_block = block_size

    # Compute padding for height to nearest multiple of h_block
    H_target = math.ceil(H / h_block) * h_block
    dh = H_target - H
    top, down = dh // 2, dh - dh // 2
    n_h = H_target // h_block

    # Compute padding for width to nearest multiple of w_block
    W_target = math.ceil(W / w_block) * w_block
    dw = W_target - W
    left, right = dw // 2, dw - dw // 2
    n_w = W_target // w_block

    # Apply reflection padding (order: W_left, W_right, H_top, H_bottom)
    data = F.pad(data, (left, right, top, down), mode='reflect')

    # Update shape after padding
    V, S, T, H_p, W_p = data.shape

    # Reshape and split along H and W into blocks
    data = data.reshape(V, S, T, n_h, h_block, n_w, w_block)
    data = data.permute(0, 1, 3, 5, 2, 4, 6)  # [V, S, n_h, n_w, T, h_block, w_block]
    data = data.reshape(V, S * n_h * n_w, T, h_block, w_block)

    padding = (top, down, left, right)
    return data, (n_h, n_w, padding)


def deblock_hw(data, n_h, n_w, padding):
    V, S_blk, T, h_block, w_block = data.shape
    top, down, left, right = padding

    S_orig = S_blk // (n_h * n_w)

    # Reshape to original split
    data = data.reshape(V, S_orig, n_h, n_w, T, h_block, w_block)
    data = data.permute(0, 1, 4, 2, 5, 3, 6)
    data = data.reshape(V, S_orig, T, n_h * h_block, n_w * w_block)

    # Remove padding
    H_p, W_p = n_h * h_block, n_w * w_block
    H = H_p - top - down
    W = W_p - left - right

    data = data[:, :, :, top:top+H, left:left+W]

    return data



def normalize_data(data, norm_type, axis):
    """
    Normalize data according to the specified normalization type.

    Args:
        data (np.ndarray): Input data array.
        norm_type (str): Type of normalization ('std', 'min_max', 'mean_range').
        axis (tuple or int): Axis or axes along which to compute statistics.

    Returns:
        tuple: (normalized_data, var_mean, var_scale)
    """
    norm_type = norm_type.lower()

    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy ndarray.")

    if norm_type == "std":
        var_mean = np.mean(data, axis=axis, keepdims=True)
        var_scale = np.std(data, axis=axis, keepdims=True)
        normalized_data = (data - var_mean) / var_scale

    elif norm_type == "min_max":
        var_min = np.min(data, axis=axis, keepdims=True)
        var_max = np.max(data, axis=axis, keepdims=True)
        var_range = var_max - var_min

        var_mean = var_min + var_range / 2
        var_scale = var_range / 2

        normalized_data = (data - var_mean) / var_scale

    elif norm_type == "mean_range":
        var_mean = np.mean(data, axis=axis, keepdims=True)
        var_max = np.max(data, axis=axis, keepdims=True)
        var_min = np.min(data, axis=axis, keepdims=True)
        var_scale = var_max - var_min

        normalized_data = (data - var_mean) / var_scale

    else:
        raise NotImplementedError(f"Normalization type '{norm_type}' is not implemented.")

    return normalized_data, var_mean, var_scale


class BaseDataset(Dataset):
    
    def __init__(self, args):
        args = deepcopy(args)
        self.data_path = args["data_path"]
        self.n_frame = args["n_frame"]
        
        self.block_size = None
        
        if type(self.n_frame) == list and len(self.n_frame) == 3:
            self.block_size = self.n_frame
            self.n_frame = self.n_frame[0]
        
        self.input_size = args.get("input_size", 256)
        self.augment_type = args.get("augment_type", {})
        self.norm_type = args.get("norm_type", "range")
        self.t_step = args.get("t_step", 1)
        self.inst_norm = args.get("inst_norm", False)
        
        
        self.random_crop = T.RandomCrop(size=(self.input_size, self.input_size))
        
        if "downsample" in self.augment_type:
            self.max_downsample = self.augment_type["downsample"]
        elif "randsample" in self.augment_type:
            self.max_downsample = self.augment_type["randsample"]
        else:
            self.max_downsample = 1
            
        self.enble_ds = True
            
        # print("augment_type", self.augment_type, "max_downsample", self.max_downsample)
            
    def check_config(self):
        self.true_shape = self.data_input.shape[-1]
        if self.true_shape/self.max_downsample<self.input_size and self.max_downsample !=1:
            print(self.dataset_name, "Warning: Size of image after downsampling is smaller than input size", "--> Downsampling Disable")
            self.enble_ds = False
            
        assert(self.data_input.shape[-3]%self.n_frame ==0)
        
    def change_temporal_correlation(self, input_data):
        if self.t_step==1:
            return input_data
        n_t = input_data.shape[-3]
        
        keep_frames =  n_t // (self.t_step * self.n_frame) * (self.t_step * self.n_frame)
        
        print("Change temporal correlation, keep frames   %d -> %d" %(n_t, keep_frames))
        
        input_data = input_data[..., 0:keep_frames, :, :]
        shape = input_data.shape

        input_data = input_data.reshape(*shape[:-3],  keep_frames // self.t_step, self.t_step, *shape[-2:])
        
        axis_order = list(range(len(input_data.shape) - 4)) + [-3, -4, -2, -1]
        
        input_data = input_data.transpose(axis_order)
        input_data = input_data.reshape(shape)
        
        return input_data
    
    def blocking_data(self, data):
        if self.block_size is not None:
            T, H, W = data.shape[-3:]
            b_t, b_h, b_w = self.block_size
            print(data.shape[-3:], self.block_size)
            assert(T%b_t ==0  and H%b_h ==0  and W%b_w ==0)
            
            n_t = T//b_t
            n_h = H//b_h
            n_w = W//b_w
#                                0   1   2     3    4    5    6
            data = data.reshape(-1, n_t, b_t, n_h, b_h, n_w, b_w)
            data = data.transpose([0,1,3,5,2,4,6]).reshape(-1, b_t, b_h, b_w)
            self.true_shape = b_w
            print("Data is reshaped to:", data.shape)
        else:
            data = data.reshape([-1, self.n_frame, *data.shape[-2:]])
        return data
    
    
    def uniform_data_preprocessing(self):
        
        self.data_input = self.change_temporal_correlation(self.data_input)
        self.check_config()
        
    
    def apply_augments(self,data):
        
        if ("downsample" in self.augment_type) and self.enble_ds:
            data = self.apply_downsampling(data, step=self.augment_type["downsample"])
        elif ("randsample" in self.augment_type) and self.enble_ds:
            step = torch.randint(1, self.augment_type["randsample"] + 1, (1,)).item()
            data = self.apply_downsampling(data, step=step)
            
        return data
    
    def output_processing(self, data):
        data = self.apply_augments(data)
       
        data = self.apply_padding_or_crop(data)
        offset = 0
        scale = 1
        
        if self.inst_norm:
            data, offset, scale = self.apply_inst_norm(data, True)
            
        if "rand_mask" in self.augment_type:
            size, n = self.augment_type["rand_mask"]
            masked_data = self.random_mask(data, size, n)
            data_dict = {"input":masked_data, "label":data, "offset":offset, "scale":scale}
        else:
            data_dict = {"input":data, "label":data, "offset":offset, "scale":scale}
            
        return data_dict
            
        
    def apply_downsampling(self, data, step = 1, t_step=1):
        if step==1 and t_step == 1:
            return data
        if t_step != 1:
            data = data[..., ::t_step , :,:]
        if step != 1:
            data = data[..., ::step , ::step]
        return data
    
    def random_mask(self,data, size, n):
        data = deepcopy(data)  # Avoid modifying the original data
        *leading_dims, height, width = data.shape
        if size > height or size > width:
            raise ValueError("Mask size exceeds the dimensions of the input data.")

        # Flatten leading dimensions to apply masks uniformly
        reshaped_data = data.reshape(-1, height, width)

        # Apply random masks for each "block" in the flattened data
        for block in reshaped_data:
            rows = np.random.randint(0, height - size + 1, n)
            cols = np.random.randint(0, width - size + 1, n)
            for r, c in zip(rows, cols):
                block[r:r + size, c:c + size] = 0

        return reshaped_data.reshape(data.shape)
        
        
    
    def apply_padding_or_crop(self, data):
        cur_size = data.shape[-1]
        
        if self.input_size > cur_size:
            pad_size = self.input_size - cur_size
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left
            data = F.pad(data[None], (pad_left, pad_right, pad_left, pad_right), mode='reflect')[0]
            
        elif self.input_size < cur_size:
            data = self.random_crop(data)
            
        return data
    
    def apply_inst_norm(self, data, return_norm=False):
        if self.norm_type == "mean_range":
            offset = torch.mean(data, keepdim=True)
            scale = data.max(keepdim=True).values - data.min(keepdim=True).values
            assert torch.all(scale != 0), "Scale is zero in some elements."
            data = (data - offset) / scale

        elif self.norm_type == "min_max":
            dmin = data.min(keepdim=True).values
            dmax = data.max(keepdim=True).values
            offset = (dmax + dmin) / 2
            scale = (dmax - dmin) / 2
            assert torch.all(scale != 0), "Scale is zero in some elements."
            data = (data - offset) / scale

        elif self.norm_type == "mean_range_hw":
            offset = torch.mean(data, dim=(-2, -1), keepdim=True)
            scale = torch.amax(data, dim=(-2, -1), keepdim=True) - torch.amin(data, dim=(-2, -1), keepdim=True)
            assert torch.all(scale != 0), "Scale is zero in some elements."
            data = (data - offset) / scale

        else:
            raise NotImplementedError(f"Normalization type {self.norm_type} not implemented.")

        if return_norm:
            return data, offset, scale
        else:
            return data