import os
import nibabel as nib
import torch
import numpy as np
from tqdm import tqdm

def window_image(x, min_hu=-40, max_hu=120):
    x = np.clip(x, min_hu, max_hu)
    x = (x - min_hu) / (max_hu - min_hu)
    return x

def split_to_patches(img, mask):
    patches, masks = [], []
    coords = [(0,0), (0,256), (256,0), (256,256)]
    for (i,j) in coords:
        patches.append(img[i:i+256, j:j+256].copy())
        masks.append(mask[i:i+256, j:j+256].copy())
    return patches, masks

def slice_and_save(image_folder, mask_folder, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'masks'), exist_ok=True)

    img_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.nii.gz')])
    mask_paths = sorted([os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.endswith('.nii.gz')])

    idx = 0
    for img_path, mask_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths)):
        vol_img = nib.load(img_path).get_fdata()
        vol_mask = nib.load(mask_path).get_fdata()
        D = vol_img.shape[2]

        for z in range(D):
            slice_img = vol_img[:, :, z]
            slice_mask = vol_mask[:, :, z]

            slice_img = window_image(slice_img)

            if slice_img.shape != (512, 512):
                continue

            img_patches, mask_patches = split_to_patches(slice_img, slice_mask)

            for ip, mp in zip(img_patches, mask_patches):
                # Skip pure background patches
                if np.sum(mp) == 0:
                    continue
                torch.save(torch.tensor(ip).float().unsqueeze(0), os.path.join(save_folder, 'images', f'{idx}.pt'))
                torch.save(torch.tensor(mp).float().unsqueeze(0), os.path.join(save_folder, 'masks', f'{idx}.pt'))
                idx += 1

    print(f"Saved {idx} image-mask patch pairs.")

# Example usage
if __name__ == "__main__":
    image_folder = "BHSD/label_192/images"
    mask_folder = "BHSD/label_192/ground truths"
    save_folder = "BHSD/preprocessed_patches"

    slice_and_save(image_folder, mask_folder, save_folder)
