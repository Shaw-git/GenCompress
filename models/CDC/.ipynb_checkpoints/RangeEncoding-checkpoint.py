import math
import torch
import numpy as np
from compressai.entropy_models import GaussianConditional
# from compressai.entropy_models.entropy_models_vbr import EntropyModelVbr


def _build_indexes(size):
    dims = len(size)
    N = size[0]
    C = size[1]

    view_dims = np.ones((dims,), dtype=np.int64)
    view_dims[1] = -1
    indexes = torch.arange(C).view(*view_dims)
    indexes = indexes.int()

    return indexes.repeat(N, 1, *size[2:])

def _extend_ndims(tensor, n):
    return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    
class RangeCoder:
    def __init__(self, scale_bound=0.1, max_scale=20, scale_steps=128, device="cpu"):
        self.device = device
        self.gaussian = GaussianConditional(None, scale_bound=scale_bound)
        
        lower = self.gaussian.lower_bound_scale.bound.item()
        scale_table = torch.exp(torch.linspace(math.log(lower), math.log(max_scale), steps=scale_steps))
        self.gaussian.update_scale_table(scale_table)
        self.gaussian.update()
        
        
        # self.hyper_entropy = EntropyModelVbr()
        

    def compress(self, latent, mean, scale):
        latent, mean, scale = latent.to(self.device), mean.to(self.device), scale.to(self.device)
        indexes = self.gaussian.build_indexes(scale.clamp(min=0.1))
        strings = self.gaussian.compress(latent, indexes, means=mean)
        return strings
    
    def compress_hyperlatent(self, latent, medians, qs= None):
        self.hy_size= latent.size
        indexes = _build_indexes(latent.size())
        medians = medians.detach()
        spatial_dims = len(latent.size()) - 2
        medians = _extend_ndims(medians, spatial_dims)
        medians = medians.expand(latent.size(0), *([-1] * (spatial_dims + 1)))
        
        return self.gaussian.compress(latent, indexes, medians)
    
    def decompress_hyperlatent(self, strings, medians, qs= None):
        indexes = _build_indexes(self.hy_size())
        medians = medians.detach()
        spatial_dims = len(self.hy_size()) - 2
        medians = _extend_ndims(medians, spatial_dims)
        medians = medians.expand(self.hy_size(0), *([-1] * (spatial_dims + 1)))
        return self.gaussian.decompress(strings, indexes, means=medians)
    

    def decompress(self, strings, mean, scale):
        mean, scale = mean.to(self.device), scale.to(self.device)
        indexes = self.gaussian.build_indexes(scale.clamp(min=0.1))
        decoded_latent = self.gaussian.decompress(strings, indexes, means=mean)
        return decoded_latent
    
    


# Example use:
# if __name__ == "__main__":
#     latent = torch.randn(10, 1000000) * 10
#     mean   = torch.randn(10, 1000000) * 10
#     scale  = torch.ones(10, 1000000) * 0.1

#     coder = RangeCoder(device="cpu")

#     strings = coder.compress(latent, mean, scale)
#     decoded_latent = coder.decompress(strings, mean, scale)

#     print(f"Decoded shape: {decoded_latent.shape}")
