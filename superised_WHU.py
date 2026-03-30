timport torch
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import cycle
from tqdm import tqdm
import os
import csv
import torch.nn.functional as F

from models.clip_encoder import ResNetSiameseEncoder
from models.decoder import SimpleDecoder
from models.discriminator import DomainDiscriminator
from utils.awda_loss import AWDA_Manager
from utils.metrics import CDMetrics
from data.levir_dataset import LEVIRCDDataset
from data.whu_dataset import WHUDataset

# ------------------------------------------------------------------
# 1. Setup Directories
# ------------------------------------------------------------------
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results/LEVIR_Preds', exist_ok=True)

# ------------------------------------------------------------------
# 2. Data
# ------------------------------------------------------------------
base_path = os.getcwd()

whu_train = WHUDataset(
    root_dir="datasets/WHU",
    split="train",
    return_label=True
)

whu_val = WHUDataset(
    root_dir="datasets/WHU",
    split="test",
    return_label=True
)


train_loader = DataLoader(
    whu_train,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

val_loader = DataLoader(
    whu_val,
    batch_size=8,
    shuffle=False,
    num_workers=4
)



metrics = CDMetrics(device='cuda')

@torch.no_grad()
def run_validation_whu(encoder, decoder, val_loader, device='cuda'):
    encoder.eval()
    decoder.eval()

    metrics = CDMetrics(device=device, threshold = 0.4)
    metrics.reset()

    for xa, xb, y in val_loader:
        xa = xa.to(device)
        xb = xb.to(device)
        y  = y.to(device)

        y = (y > 0).long()  

        feats = encoder(xa, xb)
        
        pred = decoder(
            feats["l1"],
            feats["l2"],
            feats["l3"],
            feats["l4"]
        )
        if pred.shape[-2:] != y.shape[-2:]:
            y = F.interpolate(
                y.float(),
                size=pred.shape[-2:],
                mode="nearest"
            ).long()

        metrics.update(pred, y.squeeze(1))
    encoder.train()
    decoder.train()

    return metrics.compute()


encoder = ResNetSiameseEncoder(pretrained=False).cuda()
decoder = SimpleDecoder(
    channels=[256, 512, 1024, 2048]
).cuda()

ckpt = torch.load("checkpoints/initial_model.pth", map_location="cuda")

encoder.load_state_dict(ckpt["encoder"])
decoder.load_state_dict(ckpt["decoder"])

print(f"Loaded model from epoch {ckpt['epoch']} | F1 = {ckpt['val_f1']:.2f}")


for p in encoder.parameters():
    p.requires_grad = False

trainable_params = list(
    p for p in encoder.parameters() if p.requires_grad
) + list(
    p for p in decoder.parameters() if p.requires_grad
)

optimizer = torch.optim.SGD(
    trainable_params,
    lr=1e-3,        # LOWER than supervised
    momentum=0.9,
    weight_decay=1e-4
)

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.softmax(logits, dim=1)[:, 1]
    targets = targets.float()

    if targets.shape[-2:] != probs.shape[-2:]:
        targets = F.interpolate(
            targets.unsqueeze(1),
            size=probs.shape[-2:],
            mode="nearest"
        ).squeeze(1)

    inter = (probs * targets).sum()
    union = probs.sum() + targets.sum()
    return 1 - (2 * inter + eps) / (union + eps)




best_f1 = 0.0
save_path = "checkpoints/supervised_whu.pth"

epochs = 50

for epoch in range(epochs):
    metrics.reset()
    fg_ratios = []

    for xa, xb, y in train_loader:
        xa = xa.cuda()
        xb = xb.cuda()
        y  = (y > 0).long().squeeze(1).cuda()

        with torch.no_grad():
            feats = encoder(xa, xb)

        pred = decoder(
            feats["l1"], feats["l2"], feats["l3"], feats["l4"]
        )

        if pred.shape[-2:] != y.shape[-2:]:
            y = F.interpolate(
                y.unsqueeze(1).float(),
                size=pred.shape[-2:],
                mode="nearest"
            ).squeeze(1).long()

        l_ce   = F.cross_entropy(pred, y)
        l_dice = dice_loss(pred, y)
        loss   = l_ce + 0.6 * l_dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        fg_ratios.append((pred.argmax(1) == 1).float().mean().detach())
        metrics.update(pred, y)

    print(f"Epoch {epoch+1}")
    print(" Train:", metrics.compute())
    print(" FG ratio:", torch.stack(fg_ratios).mean().item())

    val_metrics = run_validation_whu(
        encoder, decoder, val_loader
    )
    
    current_f1 = val_metrics["F1"]
    
    print(f"[VAL] Epoch {epoch+1}: {val_metrics}")
    
    if current_f1 > best_f1:
        best_f1 = current_f1
    
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "epoch": epoch + 1,
                "val_metrics": val_metrics,
                "best_f1": best_f1
            },
            save_path
        )
    
        print(f"✅ Saved new best supervised WHU model (F1={best_f1:.2f})")
    else:
        print(f"⏳ No improvement (best F1={best_f1:.2f})")
