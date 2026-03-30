import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
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
from models.decoder import SimpleDecoder


def evaluate_whu_with_gt(
    model_path,
    device="cuda"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # -------- Model --------
    encoder = ResNetSiameseEncoder(pretrained=False).to(device)
    decoder = SimpleDecoder(
        channels=[256, 512, 1024, 2048]
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    encoder.eval()
    decoder.eval()

    # -------- Dataset (GT AVAILABLE) --------
    ds = WHUDataset(
        root_dir="datasets/WHU-CD-256",
        return_label=True
    )
    test_indices = torch.load("splits/whu_test_indices.pt")
    
    test_ds = torch.utils.data.Subset(ds, test_indices)

    loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8
    )

    y_true_all, y_pred_all = [], []

    # -------- Inference + Eval --------
    with torch.no_grad():
        for batch in loader:
            xa, xb, y = batch[:3]   # SAFE unpack
            xa = xa.to(device)
            xb = xb.to(device)
            y  = y.squeeze().cpu().numpy()  # [H,W]
            y = (y > 0).astype(np.uint8)

            feats = encoder(xa, xb)
            logits = decoder(
                feats["l1"],
                feats["l2"],
                feats["l3"],
                feats["l4"]
            )
            logits = torch.nn.functional.interpolate(
                logits,
                size=y.shape,
                mode="bilinear",
                align_corners=False
            )

            probs = torch.softmax(logits, dim=1)[:, 1]
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

            # Resize GT if needed
            if y.shape != pred.shape:
                y = np.array(
                    Image.fromarray(y.astype(np.uint8)).resize(
                        (pred.shape[1], pred.shape[0]),
                        Image.NEAREST
                    )
                )

            y_true_all.append(y.flatten())
            y_pred_all.append(pred.flatten())

    # -------- Metrics --------
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2,2):
        TN, FP, FN, TP = cm.ravel()
    else:
        TN = FP = FN = TP = 0

    print("\n================ CONFUSION MATRIX ================")
    print(cm)

    print("\n================ METRICS (WHU) ====================")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"Precision : {precision_score(y_true, y_pred, zero_division=0)*100:.2f}%")
    print(f"Recall    : {recall_score(y_true, y_pred, zero_division=0)*100:.2f}%")
    print(f"F1-score  : {f1_score(y_true, y_pred, zero_division=0)*100:.2f}%")
    print(f"IoU       : {jaccard_score(y_true, y_pred, zero_division=0)*100:.2f}%")


if __name__ == "__main__":
    evaluate_whu_with_gt(
        model_path="checkpoints/best_awda_clip_whu.pth",
    )
