# model/cfm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossFeatureModule(nn.Module):
    def __init__(self, feat_channels=[256, 512, 1024, 2048]):
        super().__init__()
        self.fusion_layers = nn.ModuleList([
            nn.Conv2d(in_channels=c*2, out_channels=c, kernel_size=1)  # Double the channels
            for c in feat_channels
        ])

        self.mask_transform = nn.ModuleList([
            nn.Conv2d(1, c, kernel_size=1)  # Project mask to feature channels
            for c in feat_channels
        ])

    def forward(self, support_feats, query_feats, support_mask):
        fused_feats = []
        for i in range(len(support_feats)):
            fused = torch.cat([support_feats[i], query_feats[i]], dim=1) # concat channels
            fused = self.fusion_layers[i](fused)

            # Incorporate the support mask
            mask_proj = self.mask_transform[i](support_mask)  # Project mask to feature scale

            # Resize the mask to match the spatial dimensions of fused
            mask_proj = F.interpolate(mask_proj, size=fused.shape[2:], mode='bilinear', align_corners=False)

            fused = fused * mask_proj  # Mask the fused features  (or try addition)

            fused_feats.append(fused)
        return fused_feats
