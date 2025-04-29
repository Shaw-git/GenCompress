import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1)))
        avg_pool = self.fc2(avg_pool)           # Shape: [b, c, 1, 1]
        avg_pool = self.sigmoid(avg_pool)       # Shape: [b, c, 1, 1]
        
        return x * avg_pool
    
class ReturnSelf(nn.Module):
    def __init__(self, in_channels=0, reduction=0):
        super(ReturnSelf, self).__init__()
        self.reduction = reduction
    
    def forward(self, x):
        return x

class UNetWithChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, catten = False):
        super(UNetWithChannelAttention, self).__init__()
        
        # Encoder
        self.encoder1 = self._block(in_channels, 64)      # In: [b, 1, 240, 240], Out: [b, 64, 240, 240]
        self.encoder2 = self._block(64, 96, stride=2)    # In: [b, 64, 240, 240], Out: [b, 96, 120, 120]
        self.encoder3 = self._block(96, 128, stride=2)   # In: [b, 96, 120, 120], Out: [b, 128, 60, 60]
        self.encoder4 = self._block(128, 256, stride=2)  # In: [b, 128, 60, 60], Out: [b, 256, 30, 30]
        
        # Middle
        self.middle = self._block(256, 256, stride=2)   # In: [b, 256, 30, 30], Out: [b, 256, 15, 15]
        
        # Channel Attention
        if catten:
            self.ca4 = ChannelAttention(256)  # Applied to [b, 256, 30, 30]
            self.ca3 = ChannelAttention(128)  # Applied to [b, 128, 60, 60]
            self.ca2 = ChannelAttention(96)   # Applied to [b, 96, 120, 120]
            self.ca1 = ChannelAttention(64)   # Applied to [b, 64, 240, 240]
        else:
            self.ca4, self.ca3, self.ca2, self.ca1 = [ReturnSelf() for i in range(4)]
        
        # Decoder
        self.upconv4 = self._upconv(256, 256)  # In: [b, 256, 15, 15], Out: [b, 256, 30, 30]
        self.decoder4 = self._block(512, 256) # After concatenation: [b, 512, 30, 30], Out: [b, 256, 30, 30]
        
        self.upconv3 = self._upconv(256, 128)  # In: [b, 256, 30, 30], Out: [b, 128, 60, 60]
        self.decoder3 = self._block(256, 128) # After concatenation: [b, 256, 60, 60], Out: [b, 128, 60, 60]
        
        self.upconv2 = self._upconv(128, 96)   # In: [b, 128, 60, 60], Out: [b, 96, 120, 120]
        self.decoder2 = self._block(192, 96)  # After concatenation: [b, 192, 120, 120], Out: [b, 96, 120, 120]
        
        self.upconv1 = self._upconv(96, 64)    # In: [b, 96, 120, 120], Out: [b, 64, 240, 240]
        self.decoder1 = self._block(128, 64)  # After concatenation: [b, 128, 240, 240], Out: [b, 64, 240, 240]
        
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1) # In: [b, 64, 240, 240], Out: [b, out_channels, 240, 240]

    def _block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def _upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)  # Shape: [b, 64, 240, 240]
        enc2 = self.encoder2(enc1) # Shape: [b, 96, 120, 120]
        enc3 = self.encoder3(enc2) # Shape: [b, 128, 60, 60]
        enc4 = self.encoder4(enc3) # Shape: [b, 256, 30, 30]
        
        # Middle
        middle = self.middle(enc4) # Shape: [b, 256, 15, 15]
        dec4 = self.upconv4(middle) # Shape: [b, 256, 30, 30]
        
        dec4 = torch.cat([dec4, self.ca4(enc4)], dim=1) # Shape after concatenation: [b, 512, 30, 30]
        dec4 = self.decoder4(dec4) # Shape: [b, 256, 30, 30]
      
        dec3 = self.upconv3(dec4) # Shape: [b, 128, 60, 60]
        dec3 = torch.cat([dec3, self.ca3(enc3)], dim=1) # Shape after concatenation: [b, 256, 60, 60]
        dec3 = self.decoder3(dec3) # Shape: [b, 128, 60, 60]
        
        dec2 = self.upconv2(dec3) # Shape: [b, 96, 120, 120]
        dec2 = torch.cat([dec2, self.ca2(enc2)], dim=1) # Shape after concatenation: [b, 192, 120, 120]
        dec2 = self.decoder2(dec2) # Shape: [b, 96, 120, 120]
        
        dec1 = self.upconv1(dec2) # Shape: [b, 64, 240, 240]
        dec1 = torch.cat([dec1, self.ca1(enc1)], dim=1) # Shape after concatenation: [b, 128, 240, 240]
        dec1 = self.decoder1(dec1) # Shape: [b, 64, 240, 240]
        
        out = self.out_conv(dec1) # Shape: [b, out_channels, 240, 240]
        
        return out

# Example usage
# model = UNetWithChannelAttention(in_channels=1, out_channels=2)
# print(model)
