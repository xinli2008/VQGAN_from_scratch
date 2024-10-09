import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from discriminator import Discriminator
from vagan import VQGAN
from torch import autocast
from torch.cuda.amp import GradScaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "VQGAN Training Parameters")
    parser.add_argument("--latent_dim", type = int, default = 256, help = "latent dimension of vqgan")
    parser.add_argument("--image_size", type = int, default = 256, help = "image size for dataset")
    parser.add_argument("--num_codebook_vectors", type = int, default = 1024, 
                        help = "number of vqgan codebook vectors")
    parser.add_argument("--beta", type = float, default = 0.25, help = "commitment loss scalar")
    parser.add_argument("--image_channels", type = int, default = 3, help = "number of channels of images")
    parser.add_argument("--dataset_path", type = str, default = "./dataset", help = "path for dataset to save data")
    parser.add_argument("--device", type = str, default = "cuda", help = "which device to train the data, cuda or cpu")
    parser.add_argument("--batch_size", type = int, default = 64, help = "batch size for training model")
    parser.add_argument("--num_epochs", type = int, default = 1000, help = "number of epoches to train")
    parser.add_argument("--learning_rate", type = float, default = 2.25e-5, help = "learning rate")
    parser.add_argument("--beta1", type = float, default = 0.5, help = "adam beta parmeters")
    parser.add_argument("--beta2", type = float, default = 0.9, help = "adam beta parmeters")
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=0.2, help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.0,
                        help='Weighting factor for perceptual loss.')