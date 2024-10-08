import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class ImagePaths(Dataset):
    def __init__(self, path, size = None):
        self.size = size

        if not os.path.exists(path) or not os.path.isdir(path):
            raise ValueError(f"image path {path} does not exist or is not a directory")

        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self.length = len(self.images)

        self.transform = transforms.Compose([
            # NOTE: 这里默认使用的是双线性插值来进行缩放
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        image = self.transform(image)
        return image
    
def load_dataloader(args):
    train_data = ImagePaths(args.dataset_path, size = 128)
    train_dataloader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, drop_last = True)
    return train_dataloader

def init_weight(m):
    # 如果是全连接层（Linear）
    if isinstance(m, nn.Linear):
        # 使用均匀分布初始化权重
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He初始化适用于ReLU激活函数
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # 将偏置初始化为0
    
    # 如果是卷积层（Conv2d）
    elif isinstance(m, nn.Conv2d):
        # 使用正态分布初始化权重
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # 将偏置初始化为0

    # 如果是批量归一化层（BatchNorm2d）
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)  # 将权重初始化为1
        nn.init.zeros_(m.bias)  # 将偏置初始化为0