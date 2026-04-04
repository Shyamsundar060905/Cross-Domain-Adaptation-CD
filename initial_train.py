import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import os
from tqdm import tqdm
from models.clip_encoder import ResNetSiameseEncoder
from models.decoder import SimpleDecoder
from utils.metrics import CDMetrics
from data.levir_dataset import LEVIRCDDataset

device = "cuda:7"

dataset = LEVIRCDDataset(root_dir="datasets/LEVIR-CD256")

total_size = len(dataset)

train_ratio = 0.8
val_ratio   = 0.1
test_ratio  = 0.1

train_size = int(total_size * train_ratio)
val_size   = int(total_size * val_ratio)
test_size  = total_size - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=8)
val_loader   = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=8)
test_loader = DataLoader(test_set,batch_size = 16,shuffle=False,num_workers=8)
# ------------------------------
# Model
# ------------------------------
encoder = ResNetSiameseEncoder(pretrained=True).to(device)
decoder = SimpleDecoder().to(device)

optimizer = torch.optim.SGD(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=1e-3,
    momentum=0.9,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda e: max(0.0, (1 - e / 50)) ** 0.9
)

metrics = CDMetrics(device=device)

# ------------------------------
# Dice Loss
# ------------------------------
def dice_loss(logits, targets, eps=1e-5):
    probs = torch.softmax(logits, dim=1)[:, 1]

    if targets.shape[-2:] != probs.shape[-2:]:
        targets = F.interpolate(
            targets.unsqueeze(1).float(),
            size=probs.shape[-2:],
            mode="nearest"
        ).squeeze(1)

    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()

    
# ------------------------------
# Validation
# ------------------------------
@torch.no_grad()
def validate():
    encoder.eval()
    decoder.eval()
    metrics.reset()

    for xa, xb, y in val_loader:
        xa, xb, y = xa.to(device), xb.to(device), y.to(device)
        y = (y > 0).long().squeeze(1)

        feats = encoder(xa, xb, mode="change")
        pred = decoder(
            feats["stem"],
            feats["l1"],
            feats["l2"],
            feats["l3"],
            feats["l4"],
            task="cd"
        )

        metrics.update(pred, y)

    encoder.train()
    decoder.train()
    return metrics.compute()

# ------------------------------
# Training Loop
# ------------------------------
epochs = 100
best_f1 = 0.0
patience = 15
patience_counter = 0

# for epoch in range(epochs):
#     metrics.reset()
#     epoch_loss = 0.0

#     pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

#     for xa, xb, y in pbar:
#         xa, xb, y = xa.to(device), xb.to(device), y.to(device)
#         y = (y > 0).long().squeeze(1)

#         feats = encoder(xa, xb, mode="change")
#         pred = decoder(
#             feats["stem"],
#             feats["l1"],
#             feats["l2"],
#             feats["l3"],
#             feats["l4"],
#             task="cd"
#         )

#         loss = F.cross_entropy(pred, y) + 0.8 * dice_loss(pred, y)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         metrics.update(pred, y)
#         epoch_loss += loss.item()

#         # update progress bar
#         pbar.set_postfix({
#             "loss": epoch_loss / (pbar.n + 1),
#         })

#     scheduler.step()

#     train_metrics = metrics.compute()
#     val_metrics = validate()

#     print(f"\nEpoch {epoch+1}")
#     print("Train:", train_metrics)
#     print("Val:", val_metrics)

#     if val_metrics["F1"] > best_f1:
#         best_f1 = val_metrics["F1"]
#         patience_counter = 0

#         torch.save(
#             {
#                 "encoder": encoder.state_dict(),
#                 "decoder": decoder.state_dict(),
#                 "epoch": epoch,
#                 "val_f1": best_f1,
#             },
#             "checkpoints/initial_model.pth"
#         )
#         print("✅ Saved initial_model.pth")

#     else:
#         patience_counter += 1

#     if patience_counter >= patience:
#         print("🛑 Early stopping")
#         break# ------------------------------
# Final Test Evaluation
# ------------------------------
@torch.no_grad()
def test():
    print("\n🚀 Running Final Test Evaluation...")
    
    checkpoint = torch.load("checkpoints/initial_model.pth", map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
    print(f"Best validation F1: {checkpoint['val_f1']:.4f}")

    encoder.eval()
    decoder.eval()
    metrics.reset()

    for xa, xb, y in test_loader:
        xa, xb, y = xa.to(device), xb.to(device), y.to(device)
        y = (y > 0).long().squeeze(1)

        feats = encoder(xa, xb,mode="change")
        pred = decoder(
            feats["stem"],
            feats["l1"],
            feats["l2"],
            feats["l3"],
            feats["l4"],
            task="cd"
        )

        metrics.update(pred, y)

    test_metrics = metrics.compute()
    print("\n🔥 Final Test Metrics:", test_metrics)
    return test_metrics


test()
