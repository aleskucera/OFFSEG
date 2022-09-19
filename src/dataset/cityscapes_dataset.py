import torch
import os
import numpy as np
import scipy.misc as m
from torch.utils import data
from torch.utils.data import DataLoader
import torch.nn as nn
import sklearn.metrics as skm
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import PIL
import cv2
import yaml

from collections import namedtuple


class CityscapesDataset(data.Dataset):
    def __init__(self, path: str, split: str, size: int, transform=None):
        self.path = path
        self.split = split
        # self.size = size
        # self.transform = transform
        #
        # self.images, self.labels = self.read_files()
        # self.generate_split()
        # self.info()


if __name__ == '__main__':
    dict_ = {}
    data = yaml.safe_load(open('../../config/dataset.yaml', 'r'))
    names = data['rugd_names']
    for i, (k, v) in enumerate(data["rugd_label_map"].items()):
        dict_[f'{i:2d}' + names[i]] = {'id': int(k), 'priseg_id': v, 'color': data['rugd_color_map'][k]}

    yaml.dump(dict_, open('../../config/rugd_dataset.yaml', 'w'))
