import torch
import torch.nn as nn

class SimpleDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 7 → 14
        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(1024 + 1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # 14 → 28
        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # 28 → 56
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 56 → 112
        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 112 → 224 (reverse first conv stride)
        self.up0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # Heads
        self.recon_head = nn.Conv2d(64, 3, 1)
        self.cd_head = nn.Conv2d(64, 2, 1)

    def forward(self, stem, l1, l2, l3, l4, task="recon"):
        x = self.up4(l4)
        x = torch.cat([x, l3], dim=1)
        x = self.conv4(x)

        x = self.up3(x)
        x = torch.cat([x, l2], dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = torch.cat([x, l1], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = self.conv1(x)

        x = self.up0(x)

        if task == "recon":
            x = torch.sigmoid(self.recon_head(x))
        else:
            x = self.cd_head(x)

        return x        
        
        # class SimpleDecoder(nn.Module):
#     def __init__(self, channels):
#         super().__init__()

#         in_channels = channels[3]  # only use d4 (deepest feature)

#         # PPM pooling layers
#         self.pool1 = nn.AdaptiveAvgPool2d(1)
#         self.pool2 = nn.AdaptiveAvgPool2d(2)
#         self.pool3 = nn.AdaptiveAvgPool2d(3)
#         self.pool4 = nn.AdaptiveAvgPool2d(6)

#         # 1x1 conv after each pool
#         self.conv1 = nn.Conv2d(in_channels, 256, 1)
#         self.conv2 = nn.Conv2d(in_channels, 256, 1)
#         self.conv3 = nn.Conv2d(in_channels, 256, 1)
#         self.conv4 = nn.Conv2d(in_channels, 256, 1)

#         # Final fusion
#         self.fuse = nn.Conv2d(in_channels + 4 * 256, 256, 1)

#         # Output layer
#         self.final = nn.Conv2d(256, 2, 1)

#     def forward(self, d1, d2, d3, d4):
#         x = d4  # only deepest feature
#         h, w = x.shape[2:]

#         p1 = F.interpolate(self.conv1(self.pool1(x)), size=(h, w), mode="bilinear", align_corners=False)
#         p2 = F.interpolate(self.conv2(self.pool2(x)), size=(h, w), mode="bilinear", align_corners=False)
#         p3 = F.interpolate(self.conv3(self.pool3(x)), size=(h, w), mode="bilinear", align_corners=False)
#         p4 = F.interpolate(self.conv4(self.pool4(x)), size=(h, w), mode="bilinear", align_corners=False)

#         out = torch.cat([x, p1, p2, p3, p4], dim=1)
#         out = self.fuse(out)

#         out = F.interpolate(out, scale_factor=32, mode="bilinear", align_corners=False)  # 7→224

#         return self.final(out)