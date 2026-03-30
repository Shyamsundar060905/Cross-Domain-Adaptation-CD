import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import cycle
from tqdm import tqdm
import os
import csv
import torch.nn.functional as F
from torch.utils.data import random_split

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

train_lever = LEVIRCDDataset(
    root_dir=os.path.join(base_path, 'datasets/LEVIR-CD256')
)

train_whu = WHUDataset(
    root_dir=os.path.join(base_path, 'datasets/WHU-CD-256'),
    return_label=False
)

os.makedirs("splits", exist_ok=True)

split_path = "splits/whu_test_indices.pt"

train_size = int(0.8 * len(train_whu))
test_size = len(train_whu) - train_size

if not os.path.exists(split_path):

    generator = torch.Generator().manual_seed(42)

    train_ds, test_ds = random_split(
        train_whu,
        [train_size, test_size],
        generator=generator
    )

    torch.save(test_ds.indices, split_path)
    print("✅ WHU split created and saved.")

else:
    test_indices = torch.load(split_path)

    train_indices = list(
        set(range(len(train_whu))) - set(test_indices)
    )

    train_ds = torch.utils.data.Subset(train_whu, train_indices)
    test_ds = torch.utils.data.Subset(train_whu, test_indices)

    print("✅ WHU split loaded from file.")

full_train = train_lever 
val_ratio = 0.1           
val_size = int(len(full_train) * val_ratio)
train_size = len(full_train) - val_size

train_set, val_set = random_split(
    full_train,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)


train_loader = DataLoader(
    train_set,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

val_loader = DataLoader(
    val_set,
    batch_size=8,
    shuffle=False,
    num_workers=4
)
whu_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)



# ---------------- Source Branch ----------------
encoder_s = ResNetSiameseEncoder(pretrained=True).cuda()
decoder_s = SimpleDecoder(channels=[256, 512, 1024, 2048]).cuda()

# ---------------- Target Branch ----------------
encoder_t = ResNetSiameseEncoder(pretrained=True).cuda()
decoder_t = SimpleDecoder(channels=[256, 512, 1024, 2048]).cuda()


discriminator = DomainDiscriminator(in_dim=2048).cuda()


ckpt = torch.load('checkpoints/initial_model.pth', map_location='cuda')

encoder_s.load_state_dict(ckpt['encoder'], strict=False)
decoder_s.load_state_dict(ckpt['decoder'], strict=False)

# initialize target branch from same source weights
encoder_t.load_state_dict(ckpt['encoder'], strict=False)
decoder_t.load_state_dict(ckpt['decoder'], strict=False)

if 'discriminator' in ckpt:
    discriminator.load_state_dict(ckpt['discriminator'], strict=False)

print("✅ Loaded initial_model.pth")


optimizer_s = optim.Adam(
    list(encoder_s.parameters()) + list(decoder_s.parameters()),
    lr=1e-4
)

optimizer_t = optim.Adam(
    list(encoder_t.parameters()) + list(decoder_t.parameters()),
    lr=1e-4
)

opt_disc = optim.Adam(discriminator.parameters(), lr=1e-4)

awda = AWDA_Manager()
metrics = CDMetrics(device='cuda')

def validate(encoder, decoder, val_loader, metrics):
    encoder.eval()
    decoder.eval()
    metrics.reset()

    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for xa, xb, y in tqdm(val_loader, desc="Validation", leave=False):
            xa, xb, y = xa.cuda(), xb.cuda(), y.cuda()

            # ---- FIX LABELS ----
            y = y.squeeze(1)          # [B, H, W]
            y = (y > 0.5).long()      # {0,1}

            f = encoder(xa, xb)
            
            pred = decoder(
                f["l1"],
                f["l2"],
                f["l3"],
                f["l4"]
            )

            # ---- FIX RESOLUTION ----
            pred = F.interpolate(
                pred,
                size=y.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            loss = F.cross_entropy(pred, y)
            val_loss += loss.item()
            num_batches += 1

            metrics.update(pred, y)

    val_metrics = metrics.compute()
    val_metrics['Loss'] = val_loss / max(1, num_batches)

    return val_metrics
def dice_loss(pred, target, eps=1e-5):
    pred = torch.softmax(pred, dim=1)[:, 1]   # foreground prob
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    return 1 - (2 * intersection + eps) / (union + eps)

# ------------------------------------------------------------------
# 4. Training setup
# ------------------------------------------------------------------
patience = 10
patience_counter = 0
max_epochs = 100
epochs = max_epochs
total_iters = epochs * len(train_loader)

best_f1 = 0.0   


for epoch in range(epochs):
    encoder_s.train()
    decoder_s.train()
    encoder_t.train()
    decoder_t.train()
    discriminator.train()
    target_iter = cycle(whu_loader)
    metrics.reset()

    for i, (xa_s, xb_s, y_s) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch}")
    ):
        curr_iter = epoch * len(train_loader) + i

        xa_t, xb_t = next(target_iter)

        xa_s, xb_s, y_s = xa_s.cuda(), xb_s.cuda(), y_s.cuda()
        xa_t, xb_t = xa_t.cuda(), xb_t.cuda()

        # ---------------- A. Supervised Loss (LEVIR) ----------------
        f_s = encoder_s(xa_s, xb_s)
        pred_s = decoder_s(f_s["l1"],f_s["l2"],f_s["l3"],f_s["l4"])


        y_s = y_s.squeeze(1)

        y_s = (y_s > 0.5).long()

        pred_s = F.interpolate(
            pred_s,
            size=y_s.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        l_sup = F.cross_entropy(pred_s, y_s)
        metrics.update(pred_s, y_s)

        # ---------------- B. Domain Adversarial Loss ----------------
        import math
        lambda_adv = 2.0 / (1.0 + math.exp(-5 * curr_iter / total_iters)) - 1.0

        f_t = encoder_t(xa_t, xb_t)
        # --- Small source supervision for target branch ---
        f_s_t = encoder_t(xa_s, xb_s)
        pred_s_t = decoder_t(
            f_s_t["l1"],
            f_s_t["l2"],
            f_s_t["l3"],
            f_s_t["l4"]
        )
        
        pred_s_t = F.interpolate(
            pred_s_t,
            size=y_s.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        l_sup_t = F.cross_entropy(pred_s_t, y_s)

        d_s = discriminator(f_s['l4'].detach(), lambda_adv)
        d_t = discriminator(f_t['l4'], lambda_adv)

        ds_label = torch.zeros(
            d_s.size(0), d_s.size(2), d_s.size(3),
            device=d_s.device
        ).long()

        dt_label = torch.ones(
            d_t.size(0), d_t.size(2), d_t.size(3),
            device=d_t.device
        ).long()

        l_dmn = (
            F.cross_entropy(d_s, ds_label) +
            F.cross_entropy(d_t, dt_label)
        )

        # ---------------- C. CWST Loss ----------------
        pred_t = decoder_t(
            f_t["l1"],
            f_t["l2"],
            f_t["l3"],
            f_t["l4"]
        )

        pred_t = F.interpolate(
            pred_t,
            size=pred_s.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        pseudo_label = torch.argmax(pred_t.detach(), dim=1)
        weights = awda.update_weights(pred_s, y_s, curr_iter, total_iters)

        l_cwst = awda.get_cwst_loss(
            pred_t, pseudo_label, weights, curr_iter, total_iters
        )

        lambda_st = 0.3 * curr_iter / total_iters
        l_dice = dice_loss(pred_s, y_s)
        loss = (
            l_sup
            + 0.6 * l_dice
            + 0.2 * l_sup_t      # small replay
            + lambda_adv * l_dmn
            + lambda_st * l_cwst
        )
        optimizer_s.zero_grad()
        optimizer_t.zero_grad()
        opt_disc.zero_grad()
        
        loss.backward()
        
        optimizer_s.step()
        optimizer_t.step()
        opt_disc.step()

    # ---------------- Epoch Metrics ----------------
    epoch_metrics = metrics.compute()
    print(f"Epoch {epoch} LEVIR-Train Metrics: {epoch_metrics}")

    # ---------------- Validation ----------------
    val_metrics = validate(encoder_s, decoder_s, val_loader, metrics)
    print(f"[VAL] Epoch {epoch} LEVIR-CD Metrics: {val_metrics}")

    # ---------------- Checkpoint ----------------
    if val_metrics['F1'] > best_f1:
        best_f1 = val_metrics['F1']
        patience_counter = 0

        torch.save({
            'encoder_s': encoder_s.state_dict(),
            'decoder_s': decoder_s.state_dict(),
            'encoder_t': encoder_t.state_dict(),
            'decoder_t': decoder_t.state_dict(),
            'discriminator': discriminator.state_dict(),
            'epoch': epoch,
            'val_metrics': val_metrics
        }, 'checkpoints/best_awda_clip_whu_separate.pth')
        print(f"✅ New best model saved (Epoch {epoch}, F1={best_f1:.2f})")

    else:
        patience_counter += 1
        print(f"⏳ No improvement for {patience_counter}/{patience}")

    if patience_counter >= patience:
        print(f"🛑 Early stopping at epoch {epoch}")
        break
