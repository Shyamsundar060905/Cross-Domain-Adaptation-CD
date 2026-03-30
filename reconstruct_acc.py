import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
import math
import torch.nn as nn
from data.whu_dataset import WHUDataset
from models.clip_encoder import ResNetSiameseEncoder

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
        

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def evaluate_reconstruction(model_path, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    encoder = ResNetSiameseEncoder(pretrained=True).to(device)
    decoder = Decoder().to(device)

    ckpt = torch.load(model_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    encoder.eval()
    decoder.eval()

    ds = WHUDataset(root_dir="datasets/WHU-CD-256", return_label=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    total_l1 = 0
    total_mse = 0
    total_ssim = 0
    total_psnr = 0
    n = 0

    with torch.no_grad():
        for T1, T2, _ in loader:
            T1 = T1.to(device)
            T2 = T2.to(device)

            # Denoising input
            T1_noisy = T1 + torch.randn_like(T1) * 0.05
            T2_noisy = T2 + torch.randn_like(T2) * 0.05

            feats = encoder(T1_noisy, T2_noisy, mode="recon")

            T1_recon = decoder(feats["stem_a"], feats["l1_a"], feats["l2_a"], feats["l3_a"], feats["l4_a"])
            T2_recon = decoder(feats["stem_b"], feats["l1_b"], feats["l2_b"], feats["l3_b"], feats["l4_b"])

            # Metrics
            l1 = F.l1_loss(T1_recon, T1)
            mse = F.mse_loss(T1_recon, T1)
            ssim_val = ssim(T1_recon, T1, data_range=1)
            psnr_val = psnr(T1_recon, T1)

            total_l1 += l1.item()
            total_mse += mse.item()
            total_ssim += ssim_val.item()
            total_psnr += psnr_val.item()
            n += 1

    print("\n===== WHU Reconstruction Metrics =====")
    print("L1   :", total_l1 / n)
    print("MSE  :", total_mse / n)
    print("SSIM :", total_ssim / n)
    print("PSNR :", total_psnr / n)

if __name__ == "__main__":
    evaluate_reconstruction(
        model_path="whu_pretrain.pth"
    )
