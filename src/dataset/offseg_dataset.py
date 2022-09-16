import os
import sys
import cv2
import torch
import numpy as np
import logging
from PIL import Image
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.utils import visualize, convert_color
from config import Config

cfg = Config()


class OFFSEGDataset(Dataset):
    def __init__(self, split=None, size=None, transform=None):

        # Check arguments
        assert split in [None, 'train', 'val', 'test']

        # Set attributes
        self.size = size
        self.split = split
        self.transform = transform

        self.std = cfg['dataset_std']
        self.mean = cfg['dataset_mean']
        self.path = cfg['dataset_dir']

        # Read files
        self.images, self.labels = self.read_files()
        self.generate_split()
        self.info()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # Read image and label
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.labels[idx], cv2.IMREAD_GRAYSCALE)

        # Apply transformations
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']

        if self.transform is None:
            img = Image.fromarray(img)

        # Normalize image
        if self.split != 'test':
            t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
            img = t(img)

        # Convert mask to tensor
        mask = torch.from_numpy(mask).long()

        return img, mask

    def info(self):
        logger = logging.getLogger(__name__)
        logger.info(f"Generated {self.split} split of size {self.__len__()}")

    def read_files(self):
        images = []
        labels = []
        for directory in os.listdir(self.path):
            seq_path = os.path.join(self.path, directory)
            if os.path.isdir(seq_path):
                seq_images, seq_labels = self._read_sequence(seq_path)
                images += seq_images
                labels += seq_labels
        images = images[:self.size]
        labels = labels[:self.size]
        return images, labels

    @staticmethod
    def _read_sequence(seq_path: str) -> (list, list):
        images = []
        labels = []
        images_path = os.path.join(seq_path, 'images')
        labels_path = os.path.join(seq_path, 'labels')
        for label_file in os.listdir(labels_path):
            if label_file.endswith('.png'):
                name = label_file.split('.')[0]
                label_path = os.path.join(labels_path, label_file)
                image_path = os.path.join(images_path, name + '.jpg')
                if not os.path.exists(image_path):
                    image_path = os.path.join(images_path, name + '.png')
                assert os.path.exists(image_path)
                images.append(image_path)
                labels.append(label_path)
        return images, labels

    def generate_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.images, self.labels, test_size=0.1,
                                                            random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15,
                                                          random_state=42)
        if self.split == 'train':
            self.images = X_train
            self.labels = y_train
        elif self.split == 'val':
            self.images = X_val
            self.labels = y_val
        elif self.split == 'test':
            self.images = X_test
            self.labels = y_test

    def show_samples(self):
        # Show 3 random samples
        for i in range(3):
            idx = np.random.randint(0, len(self.images))
            img = cv2.imread(self.images[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.labels[idx], cv2.IMREAD_GRAYSCALE)
            mask = convert_color(mask, cfg['dataset_color_map'])
            visualize(image=img, ground_truth=mask)


if __name__ == "__main__":
    dataset = OFFSEGDataset(split='train', size=1000)
    dataset.show_samples()
