# model/__init__.py
import torch
import torch.nn as nn
from .encoder import Encoder
from .cfm import CrossFeatureModule
from .sgq import SGQModule
from .decoder import PyramidDecoder

class CGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(in_channels=1)
        self.cfm = CrossFeatureModule()
        self.sgq = SGQModule()
        self.decoder = PyramidDecoder()

    def forward(self, support_img, support_mask, query_img):
        batch_size, num_shots, C, H, W = support_img.shape

        # Encode each support shot and collect multi-scale features
        support_feats_all = []
        for i in range(num_shots):
            fs = self.encoder(support_img[:, i, :, :, :])  # fs is list of features
            support_feats_all.append(fs)

        # Average support features across shots at each scale
        fused_support_feats = []
        num_scales = len(support_feats_all[0])
        for scale_idx in range(num_scales):
            # Stack features at this scale from all shots: shape [num_shots, B, C, H, W]
            stacked = torch.stack([support_feats_all[s][scale_idx] for s in range(num_shots)])
            # Average over shots dimension (dim=0)
            avg_feat = torch.mean(stacked, dim=0)
            fused_support_feats.append(avg_feat)

        # Encode query image (list of multi-scale features)
        fq = self.encoder(query_img)

        # Cross Feature Module: fuse support and query features at each scale
        fsq = self.cfm(fused_support_feats, fq, support_mask.mean(dim=1))  # mean over shots for mask

        # SGQ Module: further fuse features with spatial guidance
        fused = self.sgq(fsq, fq)

        # Decoder: decode fused features to segmentation prediction
        pred = self.decoder(fused)

        return pred
