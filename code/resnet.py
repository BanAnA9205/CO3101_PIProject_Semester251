import torch
import torch.nn as nn
import torch.nn.functional as F
from normal_init import kaiming_normal_init_, lecun_normal_init_

# A standard implementation of a Residual Block
class ResidualBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        down_sample: nn.Module | None = None,
        use_res: bool = True
    ) -> None:
        
        super(ResidualBlock, self).__init__()
        
        self.down_sample = down_sample
        self.use_res = use_res

        # dim_out = (dim_in + 2 * pad - kernel) / stride + 1
        # with these parameters, we get dim_out = dim_in
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size = 3, 
                               stride = stride,
                               padding = 1,
                               bias = False)
        self.norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size = 3, 
                               stride = 1,
                               padding = 1,
                               bias = False)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        id = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.down_sample is not None:
            id = self.down_sample(id)

        if self.use_res:
            out += id

        out = F.relu(out, inplace=True)

        return out


class ResNet18(nn.Module):
    def __init__(
        self,
        block: type[nn.Module],          # the class of the residual block (not an instance)
        blocks_per_layer: list[int],     # number of blocks in each of the 4 stages
        n_channels: int = 3,             # input channels (e.g., 3 for RGB)
        n_classes: int = 20              # number of output classes
    ) -> None:

        super(ResNet18, self).__init__()

        self.convin = nn.Conv2d(n_channels, 64, 
                               kernel_size = 7, 
                               stride = 2,
                               padding = 3,
                               bias = False)
        self.bnin = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channels = 64

        self.layer1 = self._make_layer(block, 64, blocks_per_layer[0])
        self.layer2 = self._make_layer(block, 128, blocks_per_layer[1], 2)
        self.layer3 = self._make_layer(block, 256, blocks_per_layer[2], 2)
        self.layer4 = self._make_layer(block, 512, blocks_per_layer[3], 2)

        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_classes)

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


    def _make_layer(self, block, out_channels, n_blocks, stride = 1):
        down_sample = None

        # If stride != 1 -> sample dimension reduces, or
        # If in_channels != out_channels -> number of channels increases
        # we adjust dimensions with a 1x1 conv to match the residual shape
        if stride != 1 or self.in_channels != out_channels: 
            down_sample = nn.Sequential(
                            nn.Conv2d(self.in_channels, out_channels,
                                      kernel_size = 1,
                                      stride = stride,
                                      bias = False),
                            nn.BatchNorm2d(out_channels),
                        )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, down_sample))
        self.in_channels = out_channels

        for _ in range(1, n_blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.convin(x)
        out = self.bnin(out)
        out = F.relu(out, inplace=True)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.pooler(out)
        out = torch.flatten(out, 1, -1)
        out = self.fc(out)

        return out
