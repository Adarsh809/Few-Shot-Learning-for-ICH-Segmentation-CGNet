# model/sgq.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SGQModule(nn.Module):
    def __init__(self, feat_channels=[256, 512, 1024, 2048]):
        super().__init__()
        self.fusion = nn.ModuleList([
            nn.Conv2d(2 * c, c, kernel_size=1) for c in feat_channels
        ])

    def forward(self, fsq, fq):
        fused = []
        for i in range(len(fsq)):
            f1 = fsq[i]
            f2 = fq[i]

            # Upsample/downsample f1 to match spatial size of f2 if needed
            if f1.shape[2:] != f2.shape[2:]:
                f1 = F.interpolate(f1, size=f2.shape[2:], mode='bilinear', align_corners=False)

            # Now concatenate along channel dimension
            x = torch.cat([f1, f2], dim=1)
            x = self.fusion[i](x)
            fused.append(x)
        return fused
