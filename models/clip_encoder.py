import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key   = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()

        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)

        attention = self.softmax(torch.bmm(q, k))
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out
        

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

        # Two separate attentions
        self.attn_feat = SelfAttention(2048)
        self.attn_diff = SelfAttention(2048)

    def forward(self, x_a, x_b, mode="recon"):
        # ----- Image A -----
        sa = self.stem(x_a)
        fa1 = self.layer1(sa)
        fa2 = self.layer2(fa1)
        fa3 = self.layer3(fa2)
        fa4 = self.layer4(fa3)
        fa4 = self.attn_feat(fa4)

        # ----- Image B -----
        sb = self.stem(x_b)
        fb1 = self.layer1(sb)
        fb2 = self.layer2(fb1)
        fb3 = self.layer3(fb2)
        fb4 = self.layer4(fb3)
        fb4 = self.attn_feat(fb4)

        # ----- Reconstruction mode -----
        if mode == "recon":
            return {
                "stem_a": sa, "l1_a": fa1, "l2_a": fa2, "l3_a": fa3, "l4_a": fa4,
                "stem_b": sb, "l1_b": fb1, "l2_b": fb2, "l3_b": fb3, "l4_b": fb4,
            }

        # ----- Change detection mode -----
        elif mode == "change":
            diff_stem = torch.abs(sa - sb)
            diff_l1 = torch.abs(fa1 - fb1)
            diff_l2 = torch.abs(fa2 - fb2)
            diff_l3 = torch.abs(fa3 - fb3)
            diff_l4 = torch.abs(fa4 - fb4)

            diff_l4 = self.attn_diff(diff_l4)

            return {
                "stem": diff_stem,
                "l1": diff_l1,
                "l2": diff_l2,
                "l3": diff_l3,
                "l4": diff_l4,
            }