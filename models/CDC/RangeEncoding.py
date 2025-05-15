import math
import torch
import numpy as np
from compressai.entropy_models import GaussianConditional

class RangeCoder:
    def __init__(self, scale_bound=0.1, max_scale=20, scale_steps=128, device="cpu"):
        self.device = device
        self.gaussian = GaussianConditional(None, scale_bound=scale_bound)

        lower = self.gaussian.lower_bound_scale.bound.item()
        scale_table = torch.exp(torch.linspace(math.log(lower), math.log(max_scale), steps=scale_steps))
        self.gaussian.update_scale_table(scale_table)
        self.gaussian.update()

    def compress(self, latent, mean, scale):
        latent, mean, scale = latent.to(self.device), mean.to(self.device), scale.to(self.device)
        indexes = self.gaussian.build_indexes(scale.clamp(min=0.1))
        strings = self.gaussian.compress(latent, indexes, means=mean)
        return strings

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
