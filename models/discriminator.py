import torch.nn as nn
from torch.autograd import Function

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Conv2d(128, 2, 1)
        )

    def forward(self, x, lambda_):
        x = GradientReversal.apply(x, lambda_)
        return self.net(x)   # [B, 2, H, W]
