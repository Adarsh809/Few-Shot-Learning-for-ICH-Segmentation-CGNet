# model/encoder.py
import torch.nn as nn
from torchvision.models import resnet50

class Encoder(nn.Module):
    def __init__(self, in_channels=1): # Set in_channels here
        super().__init__()
        base = resnet50(pretrained=True)

        # Modify the first convolutional layer to accept 1 channel
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),  # Changed in_channels
            base.bn1,
            base.relu,
            base.maxpool
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]  # multi-scale features
