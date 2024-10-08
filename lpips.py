import torch
import torch.nn as nn
import os
from torchvision.models import vgg16
from collections import namedtuple
import requests
from tqdm import tqdm