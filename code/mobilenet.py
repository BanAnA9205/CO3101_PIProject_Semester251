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
    

class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expansion_factor: int
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.expansion_factor = expansion_factor

        mid_channels = int(round(in_channels * expansion_factor))

        # We only use a residual connection if dimensions match
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        if expansion_factor != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU6(inplace=True)
            )
        else:
            self.expand = nn.Identity()

        # Depthwise convolution (note the groups=mid_channels parameter)
        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_channels, 
                      mid_channels, 
                      kernel_size=3, 
                      stride=stride, 
                      padding=1, 
                      groups=mid_channels, 
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )


    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)

        if self.use_residual:
            return x + out
        else:
            return out
        

class MobileNetV2(nn.Module):
    def __init__(
        self, 
        n_channels: int = 3, 
        n_classes: int = 20
    ) -> None:
        super(MobileNetV2, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            # ReLU6 = min(max(0, x), 6) -> clamps the ReLU output to be at most 6
            nn.ReLU6(inplace=True)
        )

        # Note on the parameters:
        # t: expansion factor
        # c: output channels
        # n: number of times to repeat the block
        # s: stride for the first block in the sequence
        self.layers = nn.Sequential(
            # Bottleneck 1 (t=1, c=16, n=1, s=1)
            InvertedResidual(32, 16, stride=1, expand_ratio=1),
            
            # Bottleneck 2 (t=6, c=24, n=2, s=2)
            InvertedResidual(16, 24, stride=2, expand_ratio=6),
            InvertedResidual(24, 24, stride=1, expand_ratio=6),
            
            # Bottleneck 3 (t=6, c=32, n=3, s=2)
            InvertedResidual(24, 32, stride=2, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            
            # Bottleneck 4 (t=6, c=64, n=4, s=2)
            InvertedResidual(32, 64, stride=2, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            
            # Bottleneck 5 (t=6, c=96, n=3, s=1)
            InvertedResidual(64, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
            
            # Bottleneck 6 (t=6, c=160, n=3, s=2)
            InvertedResidual(96, 160, stride=2, expand_ratio=6),
            InvertedResidual(160, 160, stride=1, expand_ratio=6),
            InvertedResidual(160, 160, stride=1, expand_ratio=6),
            
            # Bottleneck 7 (t=6, c=320, n=1, s=1)
            InvertedResidual(160, 320, stride=1, expand_ratio=6),
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        # Note: Original MobileNetV2 did not use dropout, but
        # the official PyTorch implementation includes it.
        # (see: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, n_classes)

        self._init_weights()


    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        
        x = self.pooler(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


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