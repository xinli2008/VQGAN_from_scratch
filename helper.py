import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.norm = nn.GroupNorm(
            num_groups = 32,
            num_channels=channels,
            eps=1e-6,
            affine=True
        )

    def forward(self, x):
        return self.norm(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            # NOTE: 根据卷积的计算公式, 当kernel_size = 3, stride = 1, padding = 1时, tensor的宽高不变
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor = 2.0)
        return self.conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, channels): 
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode = "constant", value = 0)
        return self.conv(x)


class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels

        self.norm = GroupNorm()
        
        # NOTE: 在自注意力机制中, W_q、W_k、W_v除了可以用nn.linear来实现, 也可以用1x1卷积来实现。 
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.projection_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        pass