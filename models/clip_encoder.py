import torch
import torch.nn as nn
from transformers import CLIPVisionModel
from torchvision.models import resnet50, ResNet50_Weights


# class CLIPSiameseEncoder(nn.Module):
#     def __init__(self, backbone="resnet50", pretrained=True, train_bn=True):
#         super().__init__()

#         if backbone == "resnet50":
#             weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
#             base = resnet50(weights=weights)
#             out_channels = 2048
#         else:
#             raise ValueError("Only resnet50 supported for now")

#         # Remove avgpool + fc
#         self.encoder = nn.Sequential(
#             base.conv1,
#             base.bn1,
#             base.relu,
#             base.maxpool,
#             base.layer1,
#             base.layer2,
#             base.layer3,
#             base.layer4,
#         )

#         self.out_channels = out_channels

#     def forward(self, x_a, x_b):
#         # Siamese forward
#         x = torch.cat([x_a, x_b], dim=0)
#         feat = self.encoder(x)          # [2B, C, H', W']
#         f_a, f_b = torch.chunk(feat, 2, dim=0)

#         # Change feature
#         return torch.abs(f_a - f_b)
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ResNetSiameseEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        net = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool
        )
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x_a, x_b, mode="recon"):
        # ----- Image A -----
        sa = self.stem(x_a)
        fa1 = self.layer1(sa)
        fa2 = self.layer2(fa1)
        fa3 = self.layer3(fa2)
        fa4 = self.layer4(fa3)

        # ----- Image B -----
        sb = self.stem(x_b)
        fb1 = self.layer1(sb)
        fb2 = self.layer2(fb1)
        fb3 = self.layer3(fb2)
        fb4 = self.layer4(fb3)

        # ----- Reconstruction mode -----
        if mode == "recon":
            return {
                "stem_a": sa, "l1_a": fa1, "l2_a": fa2, "l3_a": fa3, "l4_a": fa4,
                "stem_b": sb, "l1_b": fb1, "l2_b": fb2, "l3_b": fb3, "l4_b": fb4,
            }

        # ----- Change detection mode -----
        elif mode == "change":
            return {
                "stem": torch.abs(sa - sb),
                "l1": torch.abs(fa1 - fb1),
                "l2": torch.abs(fa2 - fb2),
                "l3": torch.abs(fa3 - fb3),
                "l4": torch.abs(fa4 - fb4),
            }