# model/decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidDecoder(nn.Module):
    def __init__(self, feat_channels=[256, 512, 1024, 2048], upscale_factor=4):  #interpolation factor should be a param
        super().__init__()
        self.up1 = nn.ConvTranspose2d(feat_channels[3], feat_channels[2], 2, stride=2)
        self.up2 = nn.ConvTranspose2d(feat_channels[2], feat_channels[1], 2, stride=2)
        self.up3 = nn.ConvTranspose2d(feat_channels[1], feat_channels[0], 2, stride=2)
        self.final = nn.Conv2d(feat_channels[0], 1, 1)
        self.upscale_factor = upscale_factor

    def forward(self, features):
        # Ensure features are accessed in the correct order
        x1, x2, x3, x4 = features[0], features[1], features[2], features[3]

        x = self.up1(x4) + x3
        x = self.up2(x) + x2
        x = self.up3(x) + x1

        x = self.final(x)

        x = F.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        return torch.sigmoid(x)
