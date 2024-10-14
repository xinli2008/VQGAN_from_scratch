import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from discriminator import Discriminator
from vqgan import VQGAN
from torch import autocast
from lpips import LPIPS
from utils import load_dataloader, init_weight
from torch import autocast


class TrainVqgan:
    def __init__(self, args):
        self.vqgan = VQGAN(args).to(device = args.device)
        self.vqgan.load_state_dict(torch.load(os.path.join("/mnt/VQGAN_from_scratch/pretrained_models", "vqgan_epoch_308_flower.pt")))
        self.discriminator = Discriminator(args).to(device = args.device)
        self.discriminator.apply(init_weight)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.optimizer_vqgan, self.optimizer_discriminator = self.set_optimizer(args)
        self.train(args)

    def set_optimizer(self, args):
        learning_rate = args.learning_rate
        optimizer_vqgan = torch.optim.Adam(params=self.vqgan.parameters(), lr = learning_rate, eps = 1e-8, betas=(args.beta1, args.beta2))
        optimizer_discriminator = torch.optim.Adam(params=self.discriminator.parameters(), lr = learning_rate, eps = 1e-8, betas=(args.beta1, args.beta2))
        return optimizer_vqgan, optimizer_discriminator

    def train(self, args):
        dataset = load_dataloader(args)
        steps_per_epoch = len(dataset)
        scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(309, args.num_epochs):
            with tqdm(range(len(dataset))) as pbar:
                for i, imgs in zip(pbar, dataset):
                    
                    ### NOTE: train discriminator
                    self.discriminator.zero_grad()
                    self.optimizer_discriminator.zero_grad()
                    with autocast(device_type="cuda", dtype = torch.float16):
                        imgs = imgs.to(device = args.device)
                        with torch.no_grad():
                            # 使用torch.no_grad()一方面是提高效率和加速显存
                            # 也是因为在训练判别器的时候, 不需要让从vqgan的结果含有梯度
                            decoded_images, codebook_indices, codebook_loss = self.vqgan(imgs)
                        disc_real = self.discriminator(imgs)
                        # 使用detach, 将decoded_images从计算图中分离出来。
                        # 如果不使用detach()的话, 判别器的损失将会计算计算生成图像的梯度,并将其传播给生成器,从而影响训练过程。
                        disc_fake = self.discriminator(decoded_images.detach())
                        d_loss_real = torch.mean(F.relu(1. - disc_real))
                        d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                        disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch * steps_per_epoch + i,
                                                              threshold = args.disc_start)
                        gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)
                    scaler.scale(gan_loss).backward()
                    scaler.step(self.optimizer_discriminator)
                    scaler.update()

                    ### NOTE: train encoder->codebook->decoder
                    self.optimizer_vqgan.zero_grad()
                    with autocast(device_type="cuda", dtype=torch.float16):
                        decoded_images, codebook_indices, codebook_loss = self.vqgan(imgs)
                        perceptual_loss = self.perceptual_loss(imgs.contiguous(), decoded_images.contiguous())
                        reconstruction_loss = torch.abs(imgs.contiguous() - decoded_images.contiguous())
                        discriminator_fake = self.discriminator(decoded_images)
                        perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * reconstruction_loss
                        perceptual_rec_loss = perceptual_rec_loss.mean()
                        g_loss = -torch.mean(discriminator_fake)
                        lbd = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                        vq_loss = perceptual_rec_loss + codebook_loss + disc_factor * lbd * g_loss
                    scaler.scale(vq_loss).backward()
                    scaler.step(self.optimizer_vqgan)
                    scaler.update()
                    
                    pbar.set_postfix(
                        Discriminator_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 5),
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 3),
                        lbd=lbd.data.cpu().numpy()
                    )
                    pbar.update(0)
                
                # save checkpoints
                if epoch % 10 == 0:
                    torch.save(self.vqgan.state_dict(), os.path.join("./checkpoints", f"vqgan_epoch_{epoch}.pt"))
                    torch.save(self.discriminator.state_dict(), os.path.join("./checkpoints", f"discriminator_epoch_{epoch}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "VQGAN Training Parameters")
    parser.add_argument("--latent_dim", type = int, default = 256, help = "latent dimension of vqgan")
    parser.add_argument("--image_size", type = int, default = 256, help = "image size for dataset")
    parser.add_argument("--num_codebook_vectors", type = int, default = 1024, 
                        help = "number of vqgan codebook vectors")
    parser.add_argument("--beta", type = float, default = 0.25, help = "commitment loss scalar")
    parser.add_argument("--image_channels", type = int, default = 3, help = "number of channels of images")
    parser.add_argument("--dataset_path", type = str, default = "./dataset", 
                        help = "path for dataset to save data")
    parser.add_argument("--vqgan_model_path", type = str, default = "./pretrained_models/vqgan_epoch_308_flower.pt", 
                        help = "path for pretrained vqgan models")
    parser.add_argument("--discriminator_model_path", type = str, default = "./pretrained_models/discriminator_epoch_308_flower.pt", help = "path for pretrained discriminator models")
    parser.add_argument("--device", type = str, default = "cuda", help = "which device to train the data, cuda or cpu")
    parser.add_argument("--batch_size", type = int, default = 16, help = "batch size for training model")
    parser.add_argument("--num_epochs", type = int, default = 1000, help = "number of epoches to train")
    parser.add_argument("--learning_rate", type = float, default = 2.25e-5, help = "learning rate")
    parser.add_argument("--beta1", type = float, default = 0.5, help = "adam beta parmeters")
    parser.add_argument("--beta2", type = float, default = 0.9, help = "adam beta parmeters")
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=0.2, help='')
    parser.add_argument('--rec_loss_factor', type=float, default=1.0, help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual_loss_factor', type=float, default=1.0,
                        help='Weighting factor for perceptual loss.')
    
    args = parser.parse_args()

    # check if the dataset path exists and is a directory
    if not os.path.exists(args.dataset_path):
        raise ValueError(f"Dataset path {args.dataset_path} does not exists")
    
    if not os.path.isdir(args.dataset_path):
        raise ValueError(f"Dataset path {args.dataset_path} is not a directory")
    
    model = TrainVqgan(args)