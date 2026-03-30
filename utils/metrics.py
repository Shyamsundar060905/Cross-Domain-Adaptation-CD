import torch
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryJaccardIndex, BinaryStatScores

class CDMetrics:
    def __init__(self, device='cuda', threshold=None):
        self.threshold = threshold

        self.precision = BinaryPrecision().to(device)
        self.recall = BinaryRecall().to(device)
        self.f1 = BinaryF1Score().to(device)
        self.iou = BinaryJaccardIndex().to(device)
        self.stats = BinaryStatScores().to(device)

    def update(self, preds, labels):
        if self.threshold is None:
            # default: argmax (what you're doing now)
            preds_idx = torch.argmax(preds, dim=1)
        else:
            # probability-based thresholding
            probs = torch.softmax(preds, dim=1)[:, 1]
            preds_idx = (probs > self.threshold).long()

        self.precision.update(preds_idx, labels)
        self.recall.update(preds_idx, labels)
        self.f1.update(preds_idx, labels)
        self.iou.update(preds_idx, labels)
        self.stats.update(preds_idx, labels)

    def compute(self):
        return {
            "Precision": self.precision.compute().item() * 100,
            "Recall": self.recall.compute().item() * 100,
            "F1": self.f1.compute().item() * 100,
            "IoU": self.iou.compute().item() * 100
        }

    def reset(self):
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.iou.reset()
        self.stats.reset()
