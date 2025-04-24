# data/dataset.py
import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from torchvision import transforms
from config import IMG_SIZE  # Assuming IMG_SIZE is defined in config.py as (height, width)

class BHSDDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir)
                           if f.endswith('.nii.gz') and '_mask' not in f]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.data_dir, image_file)
        mask_file = image_file.replace('.nii.gz', '_mask.nii.gz')
        mask_path = os.path.join(self.data_dir, mask_file)

        # Load image and mask using nibabel
        image = nib.load(image_path).get_fdata()  # shape: (512, 512, 32) - example
        mask = nib.load(mask_path).get_fdata()    # shape: (512, 512, 32) - example

        # Select a slice (e.g., the middle slice)
        slice_idx = image.shape[2] // 2  # Get the middle slice index
        image_slice = image[:, :, slice_idx]  # shape: (512, 512)
        mask_slice = mask[:, :, slice_idx]    # shape: (512, 512)

        # Normalize image to [0, 1]
        image_slice = (image_slice - np.min(image_slice)) / (np.ptp(image_slice) + 1e-8)
        image_slice = np.expand_dims(image_slice, axis=0)  # Add channel dimension: (1, 512, 512)
        mask_slice = np.expand_dims(mask_slice, axis=0)    # Add channel dimension: (1, 512, 512)

        # Resize using interpolation
        image_slice = torch.tensor(image_slice, dtype=torch.float32)
        mask_slice = torch.tensor(mask_slice, dtype=torch.float32)

        image_resized = torch.nn.functional.interpolate(image_slice.unsqueeze(0), size=IMG_SIZE, mode='bilinear', align_corners=False).squeeze(0)
        mask_resized = torch.nn.functional.interpolate(mask_slice.unsqueeze(0), size=IMG_SIZE, mode='nearest').squeeze(0)

        # Ensure mask is binary
        mask_resized = (mask_resized > 0.5).float()

        return image_resized, mask_resized

class FewShotSegDataset(Dataset):
    """
    Few-shot segmentation dataset: returns support images/masks and a query image/mask.
    """
    def __init__(self, data_dir, split='train', shot=1, way=1, transform=None, val_ratio=0.2, seed=42):
        self.bhsd_dataset = BHSDDataset(data_dir, transform)
        self.split = split
        self.shot = shot
        self.way = way
        self.val_ratio = val_ratio
        self.seed = seed

        # Split data into train/val
        np.random.seed(self.seed)
        indices = np.arange(len(self.bhsd_dataset))
        np.random.shuffle(indices)
        split_idx = int(len(indices) * (1 - self.val_ratio))
        if self.split == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Query sample
        query_idx = self.indices[idx]
        query_image, query_mask = self.bhsd_dataset[query_idx]

        # Support samples (random, without replacement, not including query)
        support_pool = list(self.indices)
        support_pool.remove(query_idx)
        if len(support_pool) < self.shot:
            # If not enough support samples, allow replacement
            support_indices = np.random.choice(support_pool, size=self.shot, replace=True)
        else:
            support_indices = np.random.choice(support_pool, size=self.shot, replace=False)

        support_images, support_masks = [], []
        for s_idx in support_indices:
            img, msk = self.bhsd_dataset[s_idx]
            support_images.append(img)
            support_masks.append(msk)

        support_images = torch.stack(support_images)  # [shot, 1, H, W]
        support_masks = torch.stack(support_masks)    # [shot, 1, H, W]

        return support_images, support_masks, query_image, query_mask
