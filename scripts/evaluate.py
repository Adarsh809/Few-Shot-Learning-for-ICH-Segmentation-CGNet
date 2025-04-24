# scripts/evaluate.py
import torch
from torch.utils.data import DataLoader
from config import *
from model import CGNet
from utils.metrics import dice_score

def evaluate():
    from data.dataset import FewShotSegDataset  # You need to implement this
    val_set = FewShotSegDataset(DATA_DIR, split='val', shot=SHOT, way=WAY)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    model = CGNet().to(DEVICE)
    model.load_state_dict(torch.load('path_to_best_model.pth'))
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in val_loader:
            support_img, support_mask, query_img, query_mask = [x.to(DEVICE) for x in batch]
            pred = model(support_img, support_mask, query_img)
            score = dice_score(pred, query_mask)
            scores.append(score.item())
    print(f"Mean Dice Score: {sum(scores)/len(scores):.4f}")

if __name__ == '__main__':
    evaluate()
