import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ----------------------
# DropPath (Stochastic Depth)
# ----------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.2):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor

# ----------------------
# Channel Attention
# ----------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.mish = nn.Mish()
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.mish(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.mish(self.fc1(self.max_pool(x))))
        return x * torch.sigmoid(avg_out + max_out)

# ----------------------
# Spatial Attention
# ----------------------
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * torch.sigmoid(self.conv(x_cat))

# ----------------------
# CBAM Block
# ----------------------
class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ----------------------
# AdaptiveConcatPool2d
# ----------------------
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], 1)

# ----------------------
# Residual Block with CBAM
# ----------------------
class ResidualCBAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop_prob=0.2):
        super(ResidualCBAMBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.mish = nn.Mish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.cbam = CBAM(out_channels)
        self.drop_path = DropPath(drop_prob)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.mish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out = self.drop_path(out)
        out += identity
        return self.mish(out)

# ----------------------
# Final Classifier
# ----------------------
class CustomCNN_CBAM_v3(nn.Module):
    def __init__(self, num_classes=3, drop_path_prob=0.2):
        super(CustomCNN_CBAM_v3, self).__init__()
        self.layer1 = ResidualCBAMBlock(1, 32, stride=2, drop_prob=drop_path_prob)
        self.layer2 = ResidualCBAMBlock(32, 64, stride=2, drop_prob=drop_path_prob)
        self.layer3 = ResidualCBAMBlock(64, 128, stride=2, drop_prob=drop_path_prob)

        self.pool = AdaptiveConcatPool2d()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.layer1(x)   # [B, 32, H/2, W/2]
        x = self.layer2(x)   # [B, 64, H/4, W/4]
        x = self.layer3(x)   # [B, 128, H/8, W/8]
        x = self.pool(x)     # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)