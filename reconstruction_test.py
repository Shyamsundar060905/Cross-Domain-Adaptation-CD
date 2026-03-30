import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import cycle
from tqdm import tqdm
import os
import csv
import torch.nn.functional as F
from torch.utils.data import random_split
import torchvision.transforms as T
import random
import math
import torch.nn as nn
from models.clip_encoder import ResNetSiameseEncoder
from models.decoder import SimpleDecoder
from models.discriminator import DomainDiscriminator
from utils.awda_loss import AWDA_Manager
from utils.metrics import CDMetrics
from data.levir_dataset import LEVIRCDDataset
from data.whu_dataset import WHUDataset

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
base_path = os.getcwd()


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up4 = nn.Conv2d(2048, 1024, 3, padding=1)
        self.up3 = nn.Conv2d(1024, 512, 3, padding=1)
        self.up2 = nn.Conv2d(512, 256, 3, padding=1)
        self.up1 = nn.Conv2d(256, 64, 3, padding=1)
        self.up0 = nn.Conv2d(64, 32, 3, padding=1)

        self.stem_proj = nn.Conv2d(64, 256, 1)  # FIX

        self.final = nn.Conv2d(32, 3, 1)

    def forward(self, stem, l1, l2, l3, l4):
        # 8 → 16
        x = F.interpolate(l4, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.up4(x)) + l3

        # 16 → 32
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.up3(x)) + l2

        # 32 → 64
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.up2(x)) + l1

        # Add stem (after projecting channels)
        stem_proj = self.stem_proj(stem)
        x = x + stem_proj

        # 64 → 128
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.up1(x))

        # 128 → 256
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.up0(x))

        return torch.sigmoid(self.final(x))
        

decoder = Decoder().to(device)
encoder = ResNetSiameseEncoder(pretrained=True).to(device)

from pytorch_msssim import ssim

def edge_loss(pred, target):
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=pred.device).view(1,1,3,3)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=pred.device).view(1,1,3,3)

    pred_gray = pred.mean(dim=1, keepdim=True)
    target_gray = target.mean(dim=1, keepdim=True)

    pred_x = F.conv2d(pred_gray, sobel_x, padding=1)
    pred_y = F.conv2d(pred_gray, sobel_y, padding=1)

    target_x = F.conv2d(target_gray, sobel_x, padding=1)
    target_y = F.conv2d(target_gray, sobel_y, padding=1)

    return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


def reconstruction_loss(encoder, decoder, T1, T2):
    T1 = T1.to(device)
    T2 = T2.to(device)

    # Denoising
    T1_noisy = T1 + torch.randn_like(T1) * 0.05
    T2_noisy = T2 + torch.randn_like(T2) * 0.05

    feats = encoder(T1_noisy, T2_noisy, mode="recon")

    T1_recon = decoder(
        feats["stem_a"], feats["l1_a"], feats["l2_a"], feats["l3_a"], feats["l4_a"]
    )

    T2_recon = decoder(
        feats["stem_b"], feats["l1_b"], feats["l2_b"], feats["l3_b"], feats["l4_b"]
    )

    # Loss components
    l1 = F.l1_loss(T1_recon, T1) + F.l1_loss(T2_recon, T2)
    mse = F.mse_loss(T1_recon, T1) + F.mse_loss(T2_recon, T2)

    ssim_loss = (1 - ssim(T1_recon, T1, data_range=1, size_average=True)) + \
                (1 - ssim(T2_recon, T2, data_range=1, size_average=True))

    edge = edge_loss(T1_recon, T1) + edge_loss(T2_recon, T2)

    # Final combined loss
    loss = l1 + 0.1 * mse + 0.5 * ssim_loss + 0.1 * edge

    return loss

checkpoint = torch.load("whu_pretrain.pth")

encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

encoder.eval()
decoder.eval()

import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    accuracy_score
)

from data.whu_dataset import WHUDataset
from models.clip_encoder import ResNetSiameseEncoder


def find_best_threshold(errors, y_true):
    best_f1 = 0
    best_t = 0

    for t in np.linspace(0.02, 0.3, 20):
        y_pred = (errors > t).astype(np.uint8)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1

# -------------------------------
# Reconstruction → Change Map
# -------------------------------
def get_change_map(T, T_recon, threshold=0.1):
    diff = torch.abs(T - T_recon).mean(dim=1)  # [B,H,W]
    change_map = (diff > threshold).int()
    return change_map

def evaluate_reconstruction_whu(model_path, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    encoder = ResNetSiameseEncoder(pretrained=True).to(device)
    decoder = Decoder().to(device)

    ckpt = torch.load(model_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    encoder.eval()
    decoder.eval()

    ds = WHUDataset(root_dir="datasets/WHU-CD-256", return_label=True)
    splits = torch.load("splits/whu_3way_split.pt")
    test_indices = splits["test"] if isinstance(splits, dict) else splits

    test_ds = torch.utils.data.Subset(ds, test_indices)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    all_errors = []
    all_gt = []

    with torch.no_grad():
        for T1, T2, y in loader:
            T1 = T1.to(device)
            T2 = T2.to(device)

            y = y.squeeze().cpu().numpy()
            y = (y > 0).astype(np.uint8)

            T1_noisy = T1 + torch.randn_like(T1) * 0.05
            T2_noisy = T2 + torch.randn_like(T2) * 0.05

            feats = encoder(T1_noisy, T2_noisy, mode="recon")

            T1_recon = decoder(feats["stem_a"], feats["l1_a"], feats["l2_a"], feats["l3_a"], feats["l4_a"])
            T2_recon = decoder(feats["stem_b"], feats["l1_b"], feats["l2_b"], feats["l3_b"], feats["l4_b"])

            T1_recon = F.interpolate(T1_recon, size=y.shape, mode="bilinear", align_corners=False)
            T2_recon = F.interpolate(T2_recon, size=y.shape, mode="bilinear", align_corners=False)

            # reconstruction error
            err1 = torch.abs(T1 - T1_recon).mean(dim=1).squeeze().cpu().numpy()
            err2 = torch.abs(T2 - T2_recon).mean(dim=1).squeeze().cpu().numpy()

            error = (err1 + err2) / 2.0

            all_errors.append(error.flatten())
            all_gt.append(y.flatten())

    errors = np.concatenate(all_errors)
    y_true = np.concatenate(all_gt)

    # -------- FIND BEST THRESHOLD --------
    best_t, best_f1 = find_best_threshold(errors, y_true)
    print("Best threshold:", best_t)
    print("Best F1:", best_f1)

    # -------- FINAL METRICS --------
    y_pred = (errors > best_t).astype(np.uint8)

    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    print("\n================ CONFUSION MATRIX ================")
    print(cm)

    print("\n================ METRICS (Reconstruction WHU) ====================")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"Precision : {precision_score(y_true, y_pred)*100:.2f}%")
    print(f"Recall    : {recall_score(y_true, y_pred)*100:.2f}%")
    print(f"F1-score  : {f1_score(y_true, y_pred)*100:.2f}%")
    print(f"IoU       : {jaccard_score(y_true, y_pred)*100:.2f}%")

if __name__ == "__main__":
    evaluate_reconstruction_whu(
        model_path="whu_pretrain.pth"
    )