# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from model import CGNet
from data.dataset import FewShotSegDataset
from config import (
    DATA_DIR, SHOT, WAY, BATCH_SIZE, NUM_EPOCHS, 
    LEARNING_RATE, MODEL_SAVE_PATH, LOG_DIR,
    IMG_SIZE, VAL_RATIO, SEED
)

# Ensure directories exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(os.path.join(LOG_DIR, f"cgnet_{timestamp}"))

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset and DataLoader
    train_set = FewShotSegDataset(
        DATA_DIR, split='train', shot=SHOT, way=WAY,
        val_ratio=VAL_RATIO, seed=SEED
    )
    val_set = FewShotSegDataset(
        DATA_DIR, split='val', shot=SHOT, way=WAY,
        val_ratio=VAL_RATIO, seed=SEED
    )
    
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    
    # Model initialization
    model = CGNet().to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_dice = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        # Training phase
        for batch_idx, (support_img, support_mask, query_img, query_mask) in enumerate(train_loader):
            # Move data to device
            support_img = support_img.to(device)
            support_mask = support_mask.to(device)
            query_img = query_img.to(device)
            query_mask = query_mask.to(device)
            
            # Forward pass
            pred = model(support_img, support_mask, query_img)
            loss = criterion(pred.squeeze(1), query_mask.squeeze(1).float())
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            epoch_loss += loss.item()
            if (batch_idx+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Calculate epoch metrics
        avg_train_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # Validation phase
        val_metrics = evaluate(model, val_loader, device)
        writer.add_scalar('Dice/val', val_metrics['dice'], epoch)
        
        # Save checkpoints
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), 
                      os.path.join(MODEL_SAVE_PATH, f'cgnet_epoch{epoch+1}.pth'))
        
        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save(model.state_dict(), 
                      os.path.join(MODEL_SAVE_PATH, 'cgnet_best.pth'))
        
        # Print epoch summary
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] => "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Dice: {val_metrics['dice']:.4f}, "
              f"Val IoU: {val_metrics['iou']:.4f}")

def evaluate(model, val_loader, device):
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for support_img, support_mask, query_img, query_mask in val_loader:
            support_img = support_img.to(device)
            support_mask = support_mask.to(device)
            query_img = query_img.to(device)
            query_mask = query_mask.to(device)

            pred = model(support_img, support_mask, query_img)
            pred_mask = (torch.sigmoid(pred) > 0.5).float()

            batch_dice = dice_coeff(pred_mask, query_mask)
            batch_iou = iou_score(pred_mask, query_mask)
            batch_acc = accuracy_score(pred_mask, query_mask)

            total_dice += batch_dice.sum().item()
            total_iou += batch_iou.sum().item()
            total_acc += batch_acc.sum().item()
            total_samples += batch_dice.numel()

    return {
        'dice': total_dice / total_samples,
        'iou': total_iou / total_samples,
        'acc': total_acc / total_samples
    }

def accuracy_score(pred, target):
    # Both pred and target are float tensors of shape [B, 1, H, W] or [B, H, W]
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    correct = (pred == target).float().sum(dim=1)
    total = pred.size(1)
    return correct / total  # Returns tensor of shape [B]


def dice_coeff(pred, target, smooth=1e-6):
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return (2. * intersection + smooth) / (union + smooth).mean()

def iou_score(pred, target, smooth=1e-6):
    intersection = (pred * target).sum(dim=(1,2,3))
    union = (pred + target - pred * target).sum(dim=(1,2,3))
    return (intersection + smooth) / (union + smooth).mean()

if __name__ == "__main__":
    main()
