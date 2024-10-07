import torch
import torch.nn as nn
import numpy as np
from encoder import GroupNorm, Swish, ResnetBlock, AttnBlock, DownsampleBlock, UpsampleBlock

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult = (1,2,4,8), num_res_block,
                 attn_resolutions, dropout = 0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end = False):
        super(Decoder, self).__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_block
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        block_in = ch*ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2**(self.resolution - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        print(f"working iwth z of shape {self.z_shape} = {np.prod(self.z_shape)} dimensions.")

        # conv_in
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, 1)

        # middle
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
        
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = UpsampleBlock(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        
        # end
        self.norm_out = GroupNorm(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = Swish(h)
        h = self.conv_out(h)
        return h