import torch
import numpy as np
from torch.utils.data import DataLoader
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
from data.levir_dataset import LEVIRCDDataset
from models.clip_encoder import ResNetSiameseEncoder
from models.decoder import SimpleDecoder


def test_cross_branches(
    model_path="checkpoints/best_awda_clip_whu_separate.pth",
    device="cuda"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ---------------- Load Models ----------------
    encoder_s = ResNetSiameseEncoder(pretrained=False).to(device)
    decoder_s = SimpleDecoder([256, 512, 1024, 2048]).to(device)

    encoder_t = ResNetSiameseEncoder(pretrained=False).to(device)
    decoder_t = SimpleDecoder([256, 512, 1024, 2048]).to(device)

    ckpt = torch.load(model_path, map_location=device)

    encoder_s.load_state_dict(ckpt["encoder_s"])
    decoder_s.load_state_dict(ckpt["decoder_s"])

    encoder_t.load_state_dict(ckpt["encoder_t"])
    decoder_t.load_state_dict(ckpt["decoder_t"])

    encoder_s.eval()
    decoder_s.eval()
    encoder_t.eval()
    decoder_t.eval()

    print("✅ Models Loaded")

    # =====================================================
    # 1️⃣ WHU TEST → SOURCE BRANCH (_s)
    # =====================================================
    whu_ds = WHUDataset(
        root_dir="datasets/WHU-CD-256",
        return_label=True
    )

    whu_test_indices = torch.load("splits/whu_test_indices.pt")
    whu_test = torch.utils.data.Subset(whu_ds, whu_test_indices)

    whu_loader = DataLoader(whu_test, batch_size=1, shuffle=False)

    evaluate_dataset(
        encoder_s,
        decoder_s,
        whu_loader,
        name="WHU on SOURCE branch (encoder_s)"
    )

    # =====================================================
    # 2️⃣ LEVIR VAL → TARGET BRANCH (_t)
    # =====================================================
    levir_ds = LEVIRCDDataset(
        root_dir="datasets/LEVIR-CD256"
    )

    # Use same split logic as training
    val_ratio = 0.1
    val_size = int(len(levir_ds) * val_ratio)
    train_size = len(levir_ds) - val_size

    _, levir_val = torch.utils.data.random_split(
        levir_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    levir_loader = DataLoader(levir_val, batch_size=1, shuffle=False)

    evaluate_dataset(
        encoder_t,
        decoder_t,
        levir_loader,
        name="LEVIR on TARGET branch (encoder_t)"
    )


# --------------------------------------------------------
# Shared evaluation logic
# --------------------------------------------------------
def evaluate_dataset(encoder, decoder, loader, name="Model"):
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for xa, xb, y in loader:
            xa = xa.cuda()
            xb = xb.cuda()

            y = y.squeeze(1).cpu().numpy()
            y = (y > 0).astype(np.uint8)

            f = encoder(xa, xb)
            pred = decoder(
                f["l1"],
                f["l2"],
                f["l3"],
                f["l4"]
            )

            pred = F.interpolate(
                pred,
                size=y.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            pred = torch.argmax(pred, dim=1).cpu().numpy()

            y_true_all.append(y.flatten())
            y_pred_all.append(pred.flatten())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    print(f"\n================ {name} =================")
    print(cm)
    print(f"Accuracy  : {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"Precision : {precision_score(y_true, y_pred, zero_division=0)*100:.2f}%")
    print(f"Recall    : {recall_score(y_true, y_pred, zero_division=0)*100:.2f}%")
    print(f"F1-score  : {f1_score(y_true, y_pred, zero_division=0)*100:.2f}%")
    print(f"IoU       : {jaccard_score(y_true, y_pred, zero_division=0)*100:.2f}%")

if __name__ == "__main__":
    test_cross_branches()
