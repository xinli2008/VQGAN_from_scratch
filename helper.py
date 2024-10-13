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


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = GroupNorm(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = GroupNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, with_conv):
        super(UpsampleBlock, self).__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor = 2.0, mode = "bilinear")
        if self.with_conv:
            return self.conv(x)
        else:
            return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class AttnBlock(nn.Module):
    def __init__(self, channels):
        super(AttnBlock, self).__init__()
        self.in_channels = channels

        self.norm = GroupNorm(channels)
        
        # NOTE: 在自注意力机制中, W_q、W_k、W_v除了可以用nn.linear来实现, 也可以用1x1卷积来实现。 
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.projection_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h = x
        h = self.norm(h)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attention_scores = torch.bmm(q, k.transpose(1, 2))
        attention_scores = attention_scores * (int(c) ** (-0.5))
        attention_scores = F.softmax(attention_scores, dim = -1)

        attention_scores = torch.bmm(attention_scores, v)
        attention_out = attention_scores.reshape(b, c, h, w)
        attention_out = self.projection_out(attention_out)

        return x + attention_out