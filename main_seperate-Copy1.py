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

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
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


encoder = ResNetSiameseEncoder(pretrained=True).to(device)
decoder = SimpleDecoder(channels=[256, 512, 1024, 2048]).to(device)
discriminator = DomainDiscriminator(in_dim=2048).to(device)

ckpt = torch.load("../checkpoints/initial_model.pth", map_location="cpu")

encoder.load_state_dict(ckpt["encoder"])

print("✅ LEVIR model loaded.")



optimizer = optim.Adam(
    list(decoder.parameters()) +
    list(encoder.parameters()),
    lr=2e-4
)
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
            xa, xb, y = xa.to(device), xb.to(device), y.to(device)

            
            y = y.squeeze(1)        
            y = (y > 0.5).long()      

            f = encoder(xa, xb, mode="change")
            
            pred = decoder(
                f["l1"],
                f["l2"],
                f["l3"],
                f["l4"]
            )

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
    pred = torch.softmax(pred, dim=1)[:, 1]  
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    return 1 - (2 * intersection + eps) / (union + eps)


l_sup_weight = 1
patience = 10
patience_counter = 0
max_epochs = 20
epochs = max_epochs
total_iters = epochs * len(train_loader)

best_f1 = 0.0   


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-6
)

scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt_disc,
    T_max=epochs,
    eta_min=1e-6
)




for epoch in range(epochs):
    disc_acc_epoch = 0
    epoch_mask_sum = 0.0
    epoch_total = 0.0

    epoch_sup = 0.0
    epoch_unsup = 0.0
    epoch_dmn = 0.0
    
    encoder.train()
    decoder.train()
    discriminator.train()
    target_iter = cycle(whu_train_loader)
    metrics.reset()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for i, (xa_s, xb_s, y_s) in enumerate(pbar):
        curr_iter = epoch * len(train_loader) + i

        xa_t, xb_t,_ = next(target_iter)

        xa_s, xb_s, y_s = xa_s.to(device), xb_s.to(device), y_s.to(device)
        xa_t, xb_t = xa_t.to(device), xb_t.to(device)
        # Supervised Loss (LEVIR) 
        f_s = encoder(xa_s, xb_s, mode="change")
        pred_s = decoder(f_s["l1"],f_s["l2"],f_s["l3"],f_s["l4"])


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



        #  Weak branch (pseudo label) 
        xa_t_w, xb_t_w = apply_augmentation(xa_t, xb_t, weak_transform)
        f_t_w = encoder(xa_t_w, xb_t_w, mode="change")
        #  Cosine Similarity
        # fs = F.adaptive_avg_pool2d(f_s['l4'], 1).flatten(1)
        # ft = F.adaptive_avg_pool2d(f_t_w['l4'], 1).flatten(1)
        
        # fs = F.normalize(fs, dim=1)
        # ft = F.normalize(ft, dim=1)
        
        # mean_fs = fs.mean(dim=0)
        # mean_ft = ft.mean(dim=0)
        
        # cos_sim = F.cosine_similarity(
        #     mean_fs.unsqueeze(0),
        #     mean_ft.unsqueeze(0)
        # ).item()

        
        pred_t_w = decoder(
            f_t_w["l1"],
            f_t_w["l2"],
            f_t_w["l3"],
            f_t_w["l4"]
        )
        
        pred_t_w = F.interpolate(
            pred_t_w,
            size=pred_s.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Confidence threshold
        prob = torch.softmax(pred_t_w.detach(), dim=1)
        max_prob, pseudo_label = torch.max(prob, dim=1)
        
        threshold = 0.95 + 0.04 * (1 - curr_iter / total_iters)        
        mask = (max_prob > threshold)     
        
        epoch_mask_sum += mask.sum().item()
        epoch_total += mask.numel()
        #  Strong branch (consistency) 
        xa_t_s, xb_t_s = apply_augmentation(xa_t, xb_t, strong_transform)
        f_t_s = encoder(xa_t_s, xb_t_s, mode="change")
        
        pred_t_s = decoder(
            f_t_s["l1"],
            f_t_s["l2"],
            f_t_s["l3"],
            f_t_s["l4"]
        )
        
        pred_t_s = F.interpolate(
            pred_t_s,
            size=pred_s.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        #  Domain Adversarial Loss 
        lambda_adv = 2.0 / (1.0 + math.exp(-10 * (curr_iter / total_iters))) - 1.0 # 0.0499 -> 0.9999

        d_s = discriminator(f_s['l4'], lambda_adv)
        d_t = discriminator(f_t_w['l4'], lambda_adv) 
        ds_label = torch.zeros_like(d_s[:,0,:,:]).long()
        dt_label = torch.ones_like(d_t[:,0,:,:]).long()
        

        with torch.no_grad():
            pred_s_domain = torch.argmax(d_s, dim=1)
            correct_s = (pred_s_domain == ds_label).float().mean()
        
            # Target predictions
            pred_t_domain = torch.argmax(d_t, dim=1)
            correct_t = (pred_t_domain == dt_label).float().mean()
        
            disc_acc = 0.5 * (correct_s + correct_t)
            disc_acc_epoch += disc_acc.item()


        l_dmn = (
            F.cross_entropy(d_s, ds_label) +
            F.cross_entropy(d_t, dt_label)
        )
        
        weights = awda.update_weights(pred_s, y_s, curr_iter, total_iters)
        
        loss_pixel = awda.get_pixelwise_loss(
            pred_t_s,
            pseudo_label,
            weights
        )
        loss_pixel = loss_pixel[mask]
        l_cwst = loss_pixel.mean()

        
        if epoch < 2:
            l_unsup_weight = 0.0
        else:
            l_unsup_weight = min(0.1, (epoch - 2) / 10)

            
        # lambda_st = 0.6 * curr_iter / total_iters
        l_dice = dice_loss(pred_s, y_s)
        l_supervised = l_sup + 0.3 * l_dice
        

        l_unsupervised = l_cwst
        
        
        loss = l_supervised + l_unsup_weight * l_unsupervised + 0.1*l_dmn
        
        epoch_dmn += (l_dmn).item()        
        
        
        epoch_sup += l_supervised.item()
        epoch_unsup += (l_unsup_weight * l_unsupervised).item()
        
        pbar.set_postfix({
            "Sup": f"{l_supervised.item():.3f}",
            "Unsup": f"{l_unsupervised.item():.3f}",
            "DMN": f"{(l_dmn).item():.3f}",       
        })        
        optimizer.zero_grad()
        opt_disc.zero_grad()
        loss.backward()
        optimizer.step()
        opt_disc.step()
    
    
    disc_acc_epoch /= len(train_loader)
    print(f"Epoch {epoch} Discriminator Accuracy: {disc_acc_epoch*100:.2f}%")

    epoch_metrics = metrics.compute()
    print(f"Epoch {epoch} LEVIR-Train Metrics: {epoch_metrics}")

    print("Avg Pseudo-label ratio:", epoch_mask_sum / epoch_total)


    print(
        f"Epoch {epoch} Avg Loss | "
        f"Sup: {epoch_sup/len(train_loader):.4f} | "
        f"Unsup: {epoch_unsup/len(train_loader):.4f} | "
        f"DMN: {epoch_dmn/len(train_loader):.4f}"
    )    #  Validation 
    val_metrics = validate(encoder, decoder, whu_val_loader, metrics)
    print(f"[VAL] Epoch {epoch} WHU-Val Metrics: {val_metrics}")    
    
    scheduler.step()
    scheduler_disc.step()

    #  Checkpoint 
    if val_metrics['F1'] > best_f1:
        best_f1 = val_metrics['F1']
        patience_counter = 0

        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'discriminator': discriminator.state_dict(),
            'epoch': epoch,
            'val_metrics': val_metrics
        }, '../checkpoints/best_seperate_awda_resnet_whu.pth')

        print(f"✅ New best model saved (Epoch {epoch}, F1={best_f1:.2f})")

    else:
        patience_counter += 1
        print(f"⏳ No improvement for {patience_counter}/{patience}")

    if patience_counter >= patience:
        print(f"🛑 Early stopping at epoch {epoch}")
        break
