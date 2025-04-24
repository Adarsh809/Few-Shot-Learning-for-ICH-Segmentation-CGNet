# config.py (updated)
import os

# Dataset
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/processed')
SHOT = 1  # K-shot learning (1-shot/5-shot)
WAY = 1   # 1-way segmentation (single class)
VAL_RATIO = 0.2  # <-- ADD THIS (20% validation split)
SEED = 42        # <-- ADD THIS (for reproducibility)

# Training
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
IMG_SIZE = (256, 256)  # Input resolution

# Paths
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
