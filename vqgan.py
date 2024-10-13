import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from codebook import Codebook

class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(in_channels=3, out_channels=128, ch_mult=[1, 1, 2, 2, 4], num_res_block=2,
                               attn_resolutions=[16], dropout=0.0, resample_with_conv=True,
                               resolution=256, z_channels=256, double_z=False).to(device = args.device)
        self.decoder = Decoder(z_channels=256, resolution=256, in_channels=3, out_ch=3, ch = 128,
                               ch_mult=[1, 1, 2, 2, 4], num_res_block=2, attn_resolutions=[16], dropout=0.0).to(device = args.device)
        self.codebook = Codebook(args).to(device = args.device)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1, 1, 0).to(device = args.device)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1, 1, 0).to(device = args.device)

    def forward(self, imgs):
        """
        Perform VQGAN forward precess
        """
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, codebook_loss = self.codebook(quant_conv_encoded_images)
        post_quant_conv_encoded_images = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_encoded_images)

        return decoded_images, codebook_indices, codebook_loss
    
    def encode(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_imagse = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, codebook_loss = self.codebook(quant_conv_encoded_imagse)
        return codebook_mapping, codebook_indices, codebook_loss
    
    def decode(self, z):
        post_quant_decoded_imagse = self.post_quant_conv(z)
        decoded_imagse = self.decoder(post_quant_decoded_imagse)
        return decoded_imagse
    
    def calculate_lambda(self, perceptual_loss, gan_loss):
        """
        Functions: 动态计算一个权重参数\lambda, 这个权重用于在训练过程中平衡感知损失(perceptual loss)和GAN损失(gan loss)。
        在VQGAN的训练中, 模型会同时优化两类损失: 感知损失和GAN损失。
        而不同的损失项在训练中的重要性可能会变化。为了平衡这两个部分, calculate_lambda函数会根据损失的梯度动态调整权重\lambda,
        以确保感知损失和gan损失对训练的贡献在一定范围内保持合理的比例。
        """

        last_layer = self.decoder.conv_out
        last_layer_weight = last_layer.weight

        # 计算感知损失(perpetual loss)对解码器最后一层权重的梯度
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph = True)[0]

        # 计算GAN损失(gan_loss)对解码器最后一层权重的梯度
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph = True)[0]

        # 通过梯度的欧式距离计算感知损失梯度和gan损失梯度的比值
        # 这个比值\lambda将用于平衡感知损失和GAN损失的权重
        lambda1 = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)

        # 将lambda的值限制在[0, 1e4]之间, 避免其值过大或者过小, 保持数值稳定性
        lambda1 = torch.clamp(lambda1, 0, 1e4).detach()
        # NOTE: detach的作用: 从当前的计算图中分离tensor, 使得该tensor不需要参与梯度计算。也就是说, 使用了detach()后, 该tensor不再被视为需要梯度的叶子结点, 不会影响到反向传播。
        # NOTE: clamp的作用: 将tensor的元素限制在指定的最小值和最大值之间, 超出范围的将被截断到边界值。比如说clamp设置的最小值是-1,但是tensor中有一个值是-2, 则会将-2设置为-1.

        return 0.8 * lambda1
    
    def load_checkpoint(self, path):
        # NOTE: 在加载模型权重的时候, strict = True时, load_state_dict会严格检查模型的结构与加载的权重文件是否完全匹配。
        # 如果模型的state_dict存在任何一个参数不匹配的话, pytorch会抛出一个错误, 提示一个不匹配的key-value。
        self.load_state_dict(self.load(path))
    
    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor