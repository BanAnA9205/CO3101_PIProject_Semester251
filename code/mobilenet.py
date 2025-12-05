import torch
import torch.nn as nn
import torch.nn.functional as F
from normal_init import kaiming_normal_init_, lecun_normal_init_

class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1
    ) -> None:

        super(DepthwiseSeparableConv, self).__init__()
        
        # Some note on the "groups" parameter:
        # It splits the input and output channels into groups with equal size,
        # and performs convolution separately on each group.
        #
        # More specifically, if "groups" = g, then the input channels are split into g groups,
        # each with size (in_channels / g), and the output channels are also split into g groups,
        # each with size (out_channels / g). 
        # (This means both in_channels and out_channels must be divisible by g.)
        #
        # Then, convolution is performed separately on each group,
        # meaning that each group of input channels is convolved to produce 
        # the corresponding group of output channels.
        self.depthwise = nn.Conv2d(in_channels, in_channels, 
                                   kernel_size=3, 
                                   stride=stride, 
                                   padding=1, 
                                   groups=in_channels,
                                   bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, 
                                   stride=1, 
                                   padding=0,
                                   bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        # Depthwise convolution
        out = self.depthwise(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        # Pointwise convolution
        out = self.pointwise(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        return out
    

class MobileNetV1(nn.Module):
    def __init__(
        self,
        n_channels: int = 3,             # input channels (e.g., 3 for RGB)
        n_classes: int = 20              # number of output classes
    ) -> None:

        super(MobileNetV1, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.layers = nn.Sequential(
            # Block 1
            DepthwiseSeparableConv(32, 64, stride=1),
            
            # Block 2 & 3
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            
            # Block 4 & 5
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            
            # Block 6 (Transition to 512)
            DepthwiseSeparableConv(256, 512, stride=2),
            
            # The 5x Block of 512 channels
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            
            # Block 12 & 13 (Transition to 1024)
            DepthwiseSeparableConv(512, 1024, stride=2),
            DepthwiseSeparableConv(1024, 1024, stride=1),
        )

        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, n_classes)

        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming weight initialization
                kaiming_normal_init_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                # LeCun weight initialization
                lecun_normal_init_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize weight to 1 (for gammas) and bias to 0 (for betas)
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layers(x)
        
        x = self.pooler(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x