import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import albumentations

# class ImagePaths(Dataset):
#     def __init__(self, path, size = None):
#         self.size = size

#         if not os.path.exists(path) or not os.path.isdir(path):
#             raise ValueError(f"image path {path} does not exist or is not a directory")

#         self.images = [os.path.join(path, file) for file in os.listdir(path)]
#         self.length = len(self.images)

#         self.transform = transforms.Compose([
#             # NOTE: 对于transformer.Resize(), 有两个需要注意的地方:
#             # 1. 默认使用的是双线性插值
#             # 2. 如果传入的是一个值, 例如transforms.Resize(256),这意味着图像的较短边将被调整为该值,而较长边将按比例缩放,以保持图像的宽高比
#             #    当你传入两个整数(transforms.Resize((256, 512)),这意味着图像将被调整为指定的宽度和高度,不管原始图像的宽高比
#             transforms.Resize((self.size, self.size)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
#         ])

#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         image = Image.open(self.images[index]).convert("RGB")
#         image = self.transform(image)
#         return image
    
# def load_dataloader(args):
#     train_data = ImagePaths(args.dataset_path, size = args.image_size)
#     train_dataloader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, drop_last = True)
#     return train_dataloader

class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size

        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def load_dataloader(args):
    train_data = ImagePaths(args.dataset_path, size=256)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    return train_loader


# def init_weight(m):
#     # 如果是全连接层（Linear）
#     if isinstance(m, nn.Linear):
#         # 使用均匀分布初始化权重
#         nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He初始化适用于ReLU激活函数
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)  # 将偏置初始化为0
#     # 如果是卷积层（Conv2d）
#     elif isinstance(m, nn.Conv2d):
#         # 使用正态分布初始化权重
#         nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)  # 将偏置初始化为0
#     # 如果是批量归一化层（BatchNorm2d）
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.ones_(m.weight)  # 将权重初始化为1
#         nn.init.zeros_(m.bias)  # 将偏置初始化为0

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)