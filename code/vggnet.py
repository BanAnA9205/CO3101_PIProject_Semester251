import torch
import torch.nn as nn
import torch.nn.functional as F
from normal_init import kaiming_normal_init_, lecun_normal_init_

class VGGBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        n_convs: int, 
        use_batchnorm: bool = True
    ) -> None:
        
        super(VGGBlock, self).__init__()
        
        layers = []
        for i in range(n_convs):
            conv_in_channels = in_channels if i == 0 else out_channels
            layers.append(
                nn.Conv2d(conv_in_channels, out_channels, 
                          kernel_size=3, 
                          stride=1, 
                          padding=1,
                          bias=not use_batchnorm)
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)