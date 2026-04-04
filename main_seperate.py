#SHARED ENCODER AND DECODER

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

from models.clip_encoder import ResNetSiameseEncoder
from models.decoder import SimpleDecoder
from models.discriminator import DomainDiscriminator
from utils.awda_loss import AWDA_Manager
from utils.metrics import CDMetrics
from data.levir_dataset import LEVIRCDDataset
from data.whu_dataset import WHUDataset
from utils.focal_loss import FocalLoss


device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results/LEVIR_Preds', exist_ok=True)

base_path = os.getcwd()

# Datasets

train_lever = LEVIRCDDataset(
    root_dir=os.path.join(base_path, 'datasets/LEVIR-CD256')
)

train_whu = WHUDataset(
    root_dir=os.path.join(base_path, 'datasets/WHU-CD-256'),
    return_label=False
)


#Split of WHU

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


#Split of LEVIR
    
split_path_levir = "splits/levir_val_indices.pt"

full_train = train_lever
val_ratio = 0.1
val_size = int(len(full_train) * val_ratio)
train_size = len(full_train) - val_size

if not os.path.exists(split_path_levir):

    generator = torch.Generator().manual_seed(42)

    train_set, val_set = random_split(
        full_train,
        [train_size, val_size],
        generator=generator
    )

    torch.save(val_set.indices, split_path_levir)
    print("✅ LEVIR split created and saved.")

else:
    val_indices = torch.load(split_path_levir)

    train_indices = list(
        set(range(len(full_train))) - set(val_indices)
    )

    train_set = torch.utils.data.Subset(full_train, train_indices)
    val_set = torch.utils.data.Subset(full_train, val_indices)

    print("✅ LEVIR split loaded from file.")


train_loader = DataLoader(
    train_set,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    drop_last=True,
    pin_memory=True
)

val_loader = DataLoader(
    val_set,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True

)

    
whu_train_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

whu_val_loader = DataLoader(
    val_ds,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

whu_test_loader = DataLoader(
    test_ds,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

def apply_augmentation(xa, xb, transform):
    return transform(xa), transform(xb)


weak_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
])

strong_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
])

focal = FocalLoss(alpha=0.85, gamma=2.0)


decoder = SimpleDecoder().to(device)
discriminator = DomainDiscriminator(in_dim=2048).to(device)

encoder_cd = ResNetSiameseEncoder(pretrained=True).to(device)   # LEVIR CD encoder
encoder_da = ResNetSiameseEncoder(pretrained=True).to(device)   # Reconstruction encoder



ckpt = torch.load("best_recon_model.pth", map_location="cpu")
encoder_da.load_state_dict(ckpt["encoder"], strict=False)


decoder_weights = ckpt["decoder"]

decoder.load_state_dict(decoder_weights, strict=False)

print("Reconstruction encoder loaded into encoder_da")


ckpt = torch.load("checkpoints/initial_model.pth", map_location="cpu")

encoder_cd.load_state_dict(ckpt["encoder"])



print("✅ LEVIR model loaded.")

optimizer_enc = optim.Adam([
    {"params": encoder_cd.parameters(), "lr": 1e-4},
    {"params": encoder_da.parameters(), "lr": 5e-5},  # slower
    {"params": decoder.parameters(), "lr": 1e-4},
])

opt_disc = optim.Adam(discriminator.parameters(), lr=1e-4)
awda = AWDA_Manager(device)
metrics = CDMetrics(device=device)

def validate(encoder, decoder, val_loader, metrics):
    encoder.eval()
    decoder.eval()
    metrics.reset()

    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for xa, xb, y in tqdm(val_loader, desc="Validation", leave=False):
            xa = xa.to(device)
            xb = xb.to(device)
            y = y.to(device)

            # Ground truth processing
            y = y.squeeze(1)
            y = (y > 0.5).long()

            # Forward
            f = encoder(xa, xb, mode="change")
            pred = decoder(f["stem"],f["l1"], f["l2"], f["l3"], f["l4"],task="cd")

            pred = F.interpolate(
                pred,
                size=y.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            # Loss
            ce = F.cross_entropy(pred, y)
            dice = dice_loss(pred, y)
            loss = ce + 0.3 * dice

            val_loss += loss.item()
            num_batches += 1

            # Metrics (IMPORTANT: use argmax)
            metrics.update(pred.detach(), y)    
    val_metrics = metrics.compute()
    val_metrics['Loss'] = val_loss / max(1, num_batches)

    return val_metrics
    
def dice_loss(pred, target, eps=1e-5):
    pred = torch.softmax(pred, dim=1)[:, 1]  
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    return 1 - (2 * intersection + eps) / (union + eps)


l_sup_weight = 1
patience = 10
patience_counter = 0
max_epochs = 30
epochs = max_epochs
total_iters = epochs * len(train_loader)

best_f1 = 0.0   


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_enc,
    T_max=epochs,
    eta_min=1e-6
)

scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt_disc,
    T_max=epochs,
    eta_min=1e-6
)

epoch_mask_sum = 0
epoch_total = 0

for epoch in range(epochs):
    encoder_cd.train()
    encoder_da.train()
    decoder.train()
    discriminator.train()

    target_iter = cycle(whu_train_loader)
    metrics.reset()

    epoch_sup = 0.0
    epoch_unsup = 0.0
    epoch_dmn = 0.0
    disc_acc_epoch = 0
    epoch_cos = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for i, (xa_s, xb_s, y_s) in enumerate(pbar):
        curr_iter = epoch * len(train_loader) + i
        xa_t, xb_t, _ = next(target_iter)

        xa_s, xb_s, y_s = xa_s.to(device), xb_s.to(device), y_s.to(device)
        xa_t, xb_t = xa_t.to(device), xb_t.to(device)

        y_s = y_s.squeeze(1)
        y_s = (y_s > 0.5).long()

        # =========================================================
        # 1. SUPERVISED LOSS (SOURCE)
        # =========================================================
        f_s = encoder_cd(xa_s, xb_s, mode="change")
        pred_s = decoder(f_s["stem"],f_s["l1"], f_s["l2"], f_s["l3"], f_s["l4"],task="cd")

        pred_s = F.interpolate(pred_s, size=y_s.shape[-2:], mode='bilinear', align_corners=False)

        l_sup = focal(pred_s, y_s)
        l_dice = dice_loss(pred_s, y_s)
        l_supervised = l_sup + 0.3 * l_dice
        
        metrics.update(pred_s.detach(), y_s)        
        
        # =========================================================
        # 2. PSEUDO LABEL (WEAK AUG TARGET)
        # =========================================================
        xa_t_w, xb_t_w = apply_augmentation(xa_t, xb_t, weak_transform)

        f_t_w = encoder_cd(xa_t_w, xb_t_w, mode="change")


        pred_t_w = decoder(f_t_w["stem"],f_t_w["l1"], f_t_w["l2"], f_t_w["l3"], f_t_w["l4"],task="cd")
        pred_t_w = F.interpolate(pred_t_w, size=xa_t.shape[-2:], mode='bilinear', align_corners=False)

        prob = torch.softmax(pred_t_w.detach(), dim=1)
        max_prob, pseudo_label = torch.max(prob, dim=1)

        threshold = 0.95
        mask = (max_prob > threshold)
        epoch_mask_sum += mask.sum().item()
        epoch_total += mask.numel()
        # =========================================================
        # 3. STRONG AUG CONSISTENCY LOSS
        # =========================================================
        xa_t_s, xb_t_s = apply_augmentation(xa_t, xb_t, strong_transform)

        f_t_s = encoder_da(xa_t_s, xb_t_s, mode="change")
        pred_t_s = decoder(f_t_s["stem"],f_t_s["l1"], f_t_s["l2"], f_t_s["l3"], f_t_s["l4"], task="cd")
        pred_t_s = F.interpolate(pred_t_s, size=xa_t.shape[-2:], mode='bilinear', align_corners=False)

        weights = awda.update_weights(pred_s, y_s, curr_iter, total_iters)
        loss_pixel = awda.get_pixelwise_loss(pred_t_s, pseudo_label, weights)
        loss_pixel = loss_pixel[mask]

        if mask.sum() > 0:
            l_unsupervised = loss_pixel.mean()
        else:
            l_unsupervised = torch.tensor(0.0, device=device)
        # =========================================================
        # 4. DOMAIN DISCRIMINATOR TRAIN
        # =========================================================
        f_s_da = encoder_cd(xa_s, xb_s, mode="change")
        f_t_da = encoder_da(xa_t, xb_t, mode="change")
        
        d_s = discriminator(f_s_da["l4"], 0)  # no GRL
        d_t = discriminator(f_t_da["l4"], 0)
                #  Cosine Similarity
        fs = F.adaptive_avg_pool2d(f_s_da['l4'], 1).flatten(1)
        ft = F.adaptive_avg_pool2d(f_t_da['l4'], 1).flatten(1)
        
        fs = F.normalize(fs, dim=1)
        ft = F.normalize(ft, dim=1)
        
        mean_fs = fs.mean(dim=0)
        mean_ft = ft.mean(dim=0)
        
        cos_sim = F.cosine_similarity(
            mean_fs.unsqueeze(0),
            mean_ft.unsqueeze(0)
        ).item()


                
        ds_label = torch.zeros(d_s.size(0), d_s.size(2), d_s.size(3), device=device).long()
        dt_label = torch.ones(d_t.size(0), d_t.size(2), d_t.size(3), device=device).long()
        

        loss_disc = F.cross_entropy(d_s, ds_label) + F.cross_entropy(d_t, dt_label)
        
        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # discriminator accuracy
        with torch.no_grad():
            pred_s_domain = torch.argmax(d_s, dim=1)
            pred_t_domain = torch.argmax(d_t, dim=1)

            acc_s = (pred_s_domain == ds_label).float().mean()
            acc_t = (pred_t_domain == dt_label).float().mean()
            disc_acc = 0.5 * (acc_s + acc_t)
            disc_acc_epoch += disc_acc.item()

        # =========================================================
        # 5. ENCODER ADVERSARIAL LOSS (FOOL DISCRIMINATOR)
        # =========================================================
        lambda_adv = 2.0 / (1.0 + math.exp(-10 * (curr_iter / total_iters))) - 1.0

        f_s = encoder_cd(xa_s, xb_s, mode="change")
        f_t = encoder_da(xa_t, xb_t, mode="change")

        d_s = discriminator(f_s["l4"], lambda_adv)
        d_t = discriminator(f_t["l4"], lambda_adv)

        l_dmn = F.cross_entropy(d_s, ds_label) + F.cross_entropy(d_t, dt_label)

        # =========================================================
        # 6. TOTAL LOSS
        # =========================================================
        if epoch < 2:
            l_unsup_weight = 0.0
        else:
            l_unsup_weight = min(1.0, (epoch - 2) / 10)
            
        if epoch < 2:
            lambda_dmn = 0.0
        elif epoch < 5:
            lambda_dmn = 0.05
        else:
            lambda_dmn = 0.1
        loss = l_supervised + l_unsup_weight * l_unsupervised + lambda_dmn * l_dmn

        optimizer_enc.zero_grad()
        loss.backward()
        optimizer_enc.step()

        # logging
        epoch_sup += l_supervised.item()
        epoch_unsup += (l_unsup_weight * l_unsupervised).item()
        epoch_dmn += (lambda_dmn * l_dmn).item()
        epoch_cos += cos_sim
        pbar.set_postfix({
            "Sup": f"{l_supervised.item():.3f}",
            "Unsup": f"{(l_unsup_weight * l_unsupervised).item():.3f}",
            "DMN": f"{(lambda_dmn * l_dmn).item():.3f}",
        })

    print("Discriminator Acc:", disc_acc_epoch / len(train_loader))    

    epoch_metrics = metrics.compute()
    print(f"Epoch {epoch} LEVIR-Train Metrics: {epoch_metrics}")

    print("Avg Pseudo-label ratio:", epoch_mask_sum / epoch_total)


    print(
        f"Epoch {epoch} Avg Loss | "
        f"Sup: {epoch_sup/len(train_loader):.4f} | "
        f"Unsup: {epoch_unsup/len(train_loader):.4f} | "
        f"DMN: {epoch_dmn/len(train_loader):.4f}"
    )    #  Validation 
    print("Cosine Similarity:", epoch_cos / len(train_loader))

    val_metrics = validate(encoder_cd, decoder, whu_val_loader, metrics)
    val_da = validate(encoder_da, decoder, whu_val_loader, metrics)
    print(f"[VAL] Epoch {epoch} WHU-Val Metrics: {val_metrics}")    
    print(f"[VAL] Epoch {epoch} WHU-Val Metrics: {val_da}")    

    scheduler.step()
    scheduler_disc.step()

    #  Checkpoint 
    if val_metrics['F1'] > best_f1:
        best_f1 = val_metrics['F1']
        patience_counter = 0

        torch.save({
            'encoder': encoder_cd.state_dict(),
            'decoder': decoder.state_dict(),
            'discriminator': discriminator.state_dict(),
            'epoch': epoch,
            'val_metrics': val_metrics
        }, '../checkpoints/awda_resnet.pth')

        print(f"✅ New best model saved (Epoch {epoch}, F1={best_f1:.2f})")

    else:
        patience_counter += 1
        print(f"⏳ No improvement for {patience_counter}/{patience}")

    if patience_counter >= patience:
        print(f"🛑 Early stopping at epoch {epoch}")
        break
