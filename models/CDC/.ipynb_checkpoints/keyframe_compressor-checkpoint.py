import torch.nn as nn
from .network_components import ResnetBlock, FlexiblePrior, Downsample, Upsample
from .utils import quantize, NormalDistribution
import time
import torch 

class Compressor(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.dims = [channels, *map(lambda m: dim * m, dim_mults)]
        self.in_out = list(zip(self.dims[:-1], self.dims[1:]))
        self.reversed_dims = [*map(lambda m: dim * m, reverse_dim_mults), out_channels]
        self.reversed_in_out = list(zip(self.reversed_dims[:-1], self.reversed_dims[1:]))
        assert self.dims[-1] == self.reversed_dims[0]
        self.hyper_dims = [self.dims[-1], *map(lambda m: dim * m, hyper_dims_mults)]
        self.hyper_in_out = list(zip(self.hyper_dims[:-1], self.hyper_dims[1:]))
        self.reversed_hyper_dims = list(
            reversed([self.dims[-1] * 2, *map(lambda m: dim * m, hyper_dims_mults)])
        )
        self.reversed_hyper_in_out = list(
            zip(self.reversed_hyper_dims[:-1], self.reversed_hyper_dims[1:])
        )
        self.prior = FlexiblePrior(self.hyper_dims[-1])
        
        self.measure_time = False   
        self.encoding_time = 0
        self.decoding_time = 0
        
    def set_timer(self):
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
    def get_extra_loss(self):
        return self.prior.get_extraloss()

    def build_network(self):
        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])
        
    def hyperprior(self, shape, latent):
        x = latent
        for i, (conv, act) in enumerate(self.hyper_enc):
            x = conv(x)
            x = act(x)
            
        hyper_latent = x
        
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        
        x = q_hyper_latent
        
        for i, (deconv, act) in enumerate(self.hyper_dec):
            x = deconv(x)
            x = act(x)
        
        mean, scale = x.chunk(2, 1)
        latent_distribution = NormalDistribution(mean, scale.clamp(min=0.1))
        q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        
        state4bpp = {
            "latent": latent,
            "hyper_latent": hyper_latent,
            "latent_distribution": latent_distribution,
            "mean":mean,
            "scale": scale
        }
        return self.bpp(shape, state4bpp)

    def encode(self, input):
        
        if self.measure_time:
            start_t = time.time()
            
        for i, (resnet, down) in enumerate(self.enc):
            input = resnet(input)
            input = down(input)
            
        latent = input
        for i, (conv, act) in enumerate(self.hyper_enc):
            input = conv(input)
            input = act(input)
        hyper_latent = input
        
        
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        
        input_hyper = q_hyper_latent
        
        
        
        if self.measure_time:
            torch.cuda.synchronize()
            self.encoding_time+= time.time()-start_t
            start_t = time.time()
            
        for i, (deconv, act) in enumerate(self.hyper_dec):
            input_hyper = deconv(input_hyper)
            input_hyper = act(input_hyper)
      
        mean, scale = input_hyper.chunk(2, 1)
        
        latent_distribution = NormalDistribution(mean, scale.clamp(min=0.1))
        q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        
        
        state4bpp = {
            "latent": latent,
            "hyper_latent": hyper_latent,
            "latent_distribution": latent_distribution,
            "mean":mean,
            "scale": scale
        }
        
        
        if self.measure_time:
            torch.cuda.synchronize()
            self.decoding_time+= time.time()-start_t
            
     
        return q_latent, q_hyper_latent, state4bpp

    # def decode(self, input):
    #     output = []
    #     for i, (resnet, up) in enumerate(self.dec):
    #         input = resnet(input)
    #         input = up(input)
    #         output.append(input)
    #     return output[::-1]
    
    # def compress(self,input)
    
    def decode(self, input):
        # output = []
        if self.measure_time:
            start_t = time.time()
            
        for i, (resnet, up) in enumerate(self.dec):
            input = resnet(input)
            input = up(input)
        
        if self.measure_time:
            torch.cuda.synchronize()
            self.decoding_time+= time.time()-start_t
            
            # output.append(input)
        return input

    def bpp(self, shape, state4bpp):
        B, _, H, W = shape
        latent = state4bpp["latent"]
        hyper_latent = state4bpp["hyper_latent"]
        latent_distribution = state4bpp["latent_distribution"]
        
        if self.training:
            q_hyper_latent = quantize(hyper_latent, "noise")
            q_latent = quantize(latent, "noise")
        else:
            q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
            q_latent = quantize(latent, "dequantize", latent_distribution.mean)
            
        hyper_rate = -self.prior.likelihood(q_hyper_latent).log2()
        cond_rate = -latent_distribution.likelihood(q_latent).log2()
        bpp = (hyper_rate.sum(dim=(1, 2, 3)) + cond_rate.sum(dim=(1, 2, 3))) / (H * W)
        return bpp

    def forward(self, input):
        B,C,T,H,W = input.shape
        input = input.reshape([-1,1,H,W])
        q_latent, q_hyper_latent, state4bpp = self.encode(input)
        bpp = self.bpp(input.shape, state4bpp)
        output = self.decode(q_latent)
        output = output.reshape([B,C,T,H,W])
        return {
            "output": output,
            "bpp": bpp,
            "q_latent": q_latent,
            "q_hyper_latent": q_hyper_latent,
            "latent":state4bpp["latent"],
            "hyper_latent": state4bpp["hyper_latent"],
            "mean": state4bpp["mean"],
            "scale": state4bpp["scale"]
        }


class ResnetCompressor(Compressor):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        d3 = False
    ):
        super().__init__(
            dim,
            dim_mults,
            reverse_dim_mults,
            hyper_dims_mults,
            channels,
            out_channels
        )
        self.build_network()

    def build_network(self):

        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            self.enc.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, None, True if ind == 0 else False),
                        Downsample(dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            self.dec.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out if not is_last else dim_in),
                        Upsample(dim_out if not is_last else dim_in, dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if ind == 0
                        else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            self.hyper_dec.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if is_last
                        else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )
