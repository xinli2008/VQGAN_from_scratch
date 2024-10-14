import torch
import torch.nn as nn
import numpy as np
import math
from helper import *

class Encoder(nn.Module):
    def __init__(self, *, in_channels, out_channels, ch_mult = (1, 2, 4, 8), num_res_block,
                 attn_resolutions, dropout = 0.0, resample_with_conv = True, 
                 resolution, z_channels, double_z = True, **ignore_kwargs):
        # NOTE: 初始化中*的作用:
        super(Encoder, self).__init__()
        self.channels = out_channels
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_block = num_res_block
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.channels, 3, 1, 1)

        current_resolution = resolution
        in_ch_mult = (1, ) + tuple(ch_mult)

        # downblock
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = out_channels * in_ch_mult[i_level]
            block_out = out_channels * ch_mult[i_level]
            
            for i_block in range(self.num_res_block):
                block.append(ResnetBlock(in_channels=block_in, 
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if current_resolution in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            
            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions - 1:
                down.downsample = DownsampleBlock(block_in, resample_with_conv)
                current_resolution = current_resolution // 2
            
            self.down.append(down)
        
        # midblock
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                            out_channels=block_in,
                                            temb_channels=self.temb_ch,
                                            dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                            out_channels=block_in,
                                            temb_channels=self.temb_ch,
                                            dropout=dropout)
        
        # end 
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in,
                                  2 * z_channels if double_z else z_channels,
                                  3, 1, 1 )
    
    def forward(self, x):
        # timestep embedding
        temb = None

        # downblock
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_block):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions -1 :
                hs.append(self.down[i_level].downsample(hs[-1]))
        
        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h