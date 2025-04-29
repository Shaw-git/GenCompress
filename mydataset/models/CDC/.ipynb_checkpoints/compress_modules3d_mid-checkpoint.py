import torch.nn as nn
from .network_components import ResnetBlock, FlexiblePrior, Downsample, Upsample
from .utils import quantize, NormalDistribution



class Compressor(nn.Module):
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

    def get_extra_loss(self):
        return self.prior.get_extraloss()

    def build_network(self):
        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

    def encode(self, input):
        
        self.t_dim = input.shape[2]
        
        for i, (resnet, down) in enumerate(self.enc): # [b, 1, t, 256, 256]
            if i==0:
                input = input.permute(0,2,1,3,4)
                input = input.reshape(-1, *input.shape[2:]) # [b*t, 1, 256, 256]
            if i==2:
                input = input.reshape(-1, self.t_dim, *input.shape[1:])
                input = input.permute(0,2,1,3,4) # [b, c, t, h, w]
                
            input = resnet(input)
            input = down(input)
            

        input = input.permute(0,2,1,3,4)
        input = input.reshape(-1, *input.shape[2:])
        
            
        latent = input
        for i, (conv, act) in enumerate(self.hyper_enc):
            input = conv(input)
            input = act(input)
            
        hyper_latent = input
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        input = q_hyper_latent
        for i, (deconv, act) in enumerate(self.hyper_dec):
            input = deconv(input)
            input = act(input)

        mean, scale = input.chunk(2, 1)
        latent_distribution = NormalDistribution(mean, scale.clamp(min=0.1))
        q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        state4bpp = {
            "latent": latent,
            "hyper_latent": hyper_latent,
            "latent_distribution": latent_distribution,
        }
        return q_latent, q_hyper_latent, state4bpp, mean, scale
    
    def decode(self, input): # [n*t, c, h,w ] [8, 256, 16, 16]
        # output = []
        
        for i, (resnet, up) in enumerate(self.dec):
            if i==2:
                input = input.permute(0,2,1,3,4)
                input = input.reshape(-1, *input.shape[2:]) # [b*t, 1, 256, 256]
            if i==0:
                input = input.reshape(-1, self.t_dim//4, *input.shape[1:])
                input = input.permute(0,2,1,3,4) # [b, c, t, h, w]
                
            input = resnet(input)
            input = up(input)
        
        input = input.reshape(-1, self.t_dim, *input.shape[1:])
        input = input.permute(0,2,1,3,4)
        
        return input

    def bpp(self, shape, state4bpp):
        B, H, W = shape[0], shape[-2], shape[-1]
        n_pixels = shape[-3] * shape[-2] * shape[-1]
        
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
        
        hyper_rate = hyper_rate.reshape(B, -1).sum(dim=-1)
        cond_rate = cond_rate.reshape(B, -1).sum(dim=-1)
        
        bpp = (hyper_rate + cond_rate) / n_pixels
        return bpp, hyper_rate/n_pixels, cond_rate/n_pixels

    def forward(self, input):
        q_latent, q_hyper_latent, state4bpp, mean, scale = self.encode(input)
        bpp,hbpp,cbpp = self.bpp(input.shape, state4bpp)
        output = self.decode(q_latent)
        return {
            "output": output,
            "bpp": bpp,
            "hbpp":hbpp,
            "cbpp":cbpp,
            "mean": mean,
            "scale":scale,
            "latent": state4bpp["latent"] ,
            "q_latent": q_latent,
            "q_hyper_latent": q_hyper_latent,
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
            out_channels,
            d3
        )
        self.d3 = d3
        self.conv_layer =  nn.Conv3d if d3 else nn.Conv2d
        self.deconv_layer = nn.ConvTranspose3d if d3 else nn.ConvTranspose2d
        
        self.build_network()
        

    def build_network(self):

        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            d3 = self.d3 if ind>=2 else False
            self.enc.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, None, True if ind == 0 else False, d3 = d3),
                        Downsample(dim_out, d3 = d3),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            d3 = self.d3 if ind<2 else False
            self.dec.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out if not is_last else dim_in, d3 = d3),
                        Upsample(dim_out if not is_last else dim_in, dim_out, d3 = d3),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1) if ind == 0 else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            self.hyper_dec.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1) if is_last else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )
