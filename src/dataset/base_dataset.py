import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from sklearn.model_selection import train_test_split

from src.utils import color_to_mask


class BaseDataset(Dataset):
    def __init__(self, path, dataset, split='train', size=None,
                 transform=None, mean=None, std=None, color_map=None):

        # Check arguments
        assert split in ['train', 'val', 'test']
        assert dataset in ['rellis', 'cityscapes', 'rugd']

        # Attributes
        self.path = path
        self.size = size
        self.split = split
        self.dataset = dataset
        self.transform = transform
        self.color_map = color_map

        # Parameters
        self.std = mean
        self.mean = std

        # Data
        self.images = []
        self.labels = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # Read image
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read label. Labels in RUGD are in color format
        if self.dataset == 'rugd':
            mask = cv2.imread(self.labels[idx])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = color_to_mask(mask, self.color_map)
        else:
            mask = cv2.imread(self.labels[idx], cv2.IMREAD_GRAYSCALE)

        # Map labels
        mask = self.map_labels(mask, self.color_map)

        # Apply transformations
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']

        # Image object
        img = Image.fromarray(img)

        # Normalize image
        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)

        # Convert mask to tensor
        mask = torch.from_numpy(mask).long()

        return img, mask

    def generate_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.images, self.labels,
                                                            test_size=0.05, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          test_size=0.15, random_state=42)
        if self.split == 'train':
            self.images, self.labels = X_train, y_train
        elif self.split == 'val':
            self.images, self.labels = X_val, y_val
        elif self.split == 'test':
            self.images, self.labels = X_test, y_test

    @staticmethod
    def map_labels(label: np.ndarray, color_map: dict) -> np.ndarray:
        # Map label to new label
        for v in color_map.values():
            label[label == v['id']] = v['priseg_id']
        return label
