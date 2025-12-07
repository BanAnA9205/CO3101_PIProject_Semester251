import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import kaiming_normal_init_, lecun_normal_init_

class VGGBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        n_convs: int
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
                          bias=False)
            )

            layers.append(nn.BatchNorm2d(out_channels))

            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)
    

    def forward(self, x):
        return self.block(x)
    

class VGG16(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 1000) -> None:
        super(VGG16, self).__init__()

        # VGG16 Configuration:
        # Block 1: 2 convs, 64 channels
        # Block 2: 2 convs, 128 channels
        # Block 3: 3 convs, 256 channels
        # Block 4: 3 convs, 512 channels
        # Block 5: 3 convs, 512 channels

        self.features = nn.Sequential(
            VGGBlock(n_channels, 64, n_convs=2),
            VGGBlock(64, 128, n_convs=2),
            VGGBlock(128, 256, n_convs=3),
            VGGBlock(256, 512, n_convs=3),
            VGGBlock(512, 512, n_convs=3),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, n_classes)
        )

        self._init_weights()


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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