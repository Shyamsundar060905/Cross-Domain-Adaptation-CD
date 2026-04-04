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

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
base_path = os.getcwd()


split_dir = "splits"
os.makedirs(split_dir, exist_ok=True)

split_path = os.path.join(split_dir, "whu_3way_split.pt")

full_whu = WHUDataset(
    root_dir=os.path.join(base_path, 'datasets/WHU-CD-256'),
    return_label=True  
)

total_size = len(full_whu)

train_size = int(0.7 * total_size)
val_size   = int(0.1 * total_size)
test_size  = total_size - train_size - val_size

if not os.path.exists(split_path):

    generator = torch.Generator().manual_seed(42)

    train_ds, val_ds, test_ds = random_split(
        full_whu,
        [train_size, val_size, test_size],
        generator=generator
    )

    torch.save({
        "train": train_ds.indices,
        "val": val_ds.indices,
        "test": test_ds.indices
    }, split_path)

    print("✅ WHU 3-way split created.")

else:
    indices = torch.load(split_path)

    train_ds = torch.utils.data.Subset(full_whu, indices["train"])
    val_ds   = torch.utils.data.Subset(full_whu, indices["val"])
    test_ds  = torch.utils.data.Subset(full_whu, indices["test"])

    print("✅ WHU 3-way split loaded.")


whu_train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=8,pin_memory=True)
whu_val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=8,pin_memory=True)
whu_test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=8,pin_memory=True)



            
decoder = SimpleDecoder().to(device)
encoder = ResNetSiameseEncoder(pretrained=True).to(device)

import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_reconstruction(encoder, decoder, loader, optimizer, device):
    encoder.train()
    decoder.train()

    total_loss = 0

    for xa, xb, _ in tqdm(loader, desc="Train Recon"):
        xa = xa.to(device)
        xb = xb.to(device)

        # ---- Forward ----
        features = encoder(xa, xb, mode="recon")
        
        xa_hat = decoder(
            features["stem_a"],
            features["l1_a"],
            features["l2_a"],
            features["l3_a"],
            features["l4_a"],
            task="recon"   # IMPORTANT
        )
        
        xb_hat = decoder(
            features["stem_b"],
            features["l1_b"],
            features["l2_b"],
            features["l3_b"],
            features["l4_b"],
            task="recon"   # IMPORTANT
        )

        # ---- Loss ----
        loss_mse = F.mse_loss(xa_hat, xa) + F.mse_loss(xb_hat, xb)
        loss_l1  = F.l1_loss(xa_hat, xa) + F.l1_loss(xb_hat, xb)

        loss = loss_mse + 0.1 * loss_l1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


    
def validate_reconstruction(encoder, decoder, loader, device):
    encoder.eval()
    decoder.eval()

    total_loss = 0

    with torch.no_grad():
        for xa, xb, _ in loader:
            xa = xa.to(device)
            xb = xb.to(device)

            features = encoder(xa, xb, mode="recon")

            xa_hat = decoder(
                features["stem_a"],
                features["l1_a"],
                features["l2_a"],
                features["l3_a"],
                features["l4_a"],
                task="recon"
            )

            xb_hat = decoder(
                features["stem_b"],
                features["l1_b"],
                features["l2_b"],
                features["l3_b"],
                features["l4_b"],
                task="recon"
            )

            loss = (
                F.mse_loss(xa_hat, xa) +
                F.mse_loss(xb_hat, xb) +
                0.1 * (F.l1_loss(xa_hat, xa) + F.l1_loss(xb_hat, xb))
            )

            total_loss += loss.item()

    return total_loss / len(loader)

    
optimizer = torch.optim.Adam([
    {"params": encoder.parameters(), "lr": 1e-4},
    {"params": decoder.parameters(), "lr": 1e-4},
])


num_epochs = 150
best_val_loss = float("inf")


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6
)


for epoch in range(num_epochs):
    train_loss = train_reconstruction(encoder, decoder, whu_train_loader, optimizer, device)
    val_loss = validate_reconstruction(encoder, decoder, whu_val_loader, device)
    scheduler.step()  
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"LR: {current_lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        
        torch.save({
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss
        }, "best_recon_model.pth")
    
        print("Saved best encoder + decoder!")