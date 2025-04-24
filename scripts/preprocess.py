# scripts/preprocess.py
import os
import shutil

DATA_DIR = 'BHSD/label_192'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
MASKS_DIR = os.path.join(DATA_DIR, 'ground truths')
PROCESSED_DIR = 'data/processed'

def preprocess():
    # Create processed directory structure
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Get list of image files (excluding potential masks)
    image_files = [f for f in os.listdir(IMAGES_DIR) 
                  if f.endswith('.nii.gz') and '_mask' not in f]
    
    print(f"Found {len(image_files)} images to process")
    
    processed_count = 0
    for image_file in image_files:
        # Build source paths
        image_path = os.path.join(IMAGES_DIR, image_file)
        mask_path = os.path.join(MASKS_DIR, image_file)  # Same filename in masks dir
        
        # Verify existence
        if not os.path.exists(image_path):
            print(f"⚠️  Image file missing: {image_path}")
            continue
            
        if not os.path.exists(mask_path):
            print(f"⚠️  Mask file missing: {mask_path}")
            continue
            
        # Build destination paths
        dst_image = os.path.join(PROCESSED_DIR, image_file)
        dst_mask = os.path.join(PROCESSED_DIR, 
                              image_file.replace('.nii.gz', '_mask.nii.gz'))
        
        # Copy files with progress tracking
        try:
            shutil.copy(image_path, dst_image)
            shutil.copy(mask_path, dst_mask)
            processed_count += 1
        except Exception as e:
            print(f"🚨 Error copying {image_file}: {str(e)}")
            continue

    print(f"\n✅ Successfully processed {processed_count}/{len(image_files)} pairs")
    print(f"Processed data saved to: {os.path.abspath(PROCESSED_DIR)}")

if __name__ == "__main__":
    preprocess()
