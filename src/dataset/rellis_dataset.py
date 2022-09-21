import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from config import Config
from .base_dataset import BaseDataset
from src.utils import visualize, tensor_to_image, mask_to_color

cfg = Config()

RELLIS_PATH = '/home/ales/Datasets/Rellis_3D'


class RellisDataset(BaseDataset):
    def __init__(self, path, split=None, size=None, transform=None):
        self.ds_str = {'image_dir': 'pylon_camera_node',
                       'label_dir': 'pylon_camera_node_label_id',
                       'image_ext': '.jpg',
                       'label_ext': '.png'}

        super().__init__(path, 'rellis_dataset', split, size, transform)
        self.images, self.labels = self.read_files()
        self.generate_split()

    def read_files(self):
        images, labels = [], []

        # Walk through the dataset path until we find label, then find the corresponding image
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if self.ds_str['label_dir'] in root:
                    # Create paths
                    label_path = os.path.join(root, file)
                    image_path = label_path.replace(self.ds_str['label_dir'], self.ds_str['image_dir']) \
                        .replace(self.ds_str['label_ext'], self.ds_str['image_ext'])

                    # Append paths
                    labels.append(label_path)
                    images.append(image_path)

        # Resize dataset for development purposes (1/3 of the size because of 3 datasets)
        if self.size is not None:
            images = images[:self.size // 3]
            labels = labels[:self.size // 3]
        return images, labels


def rellis_demo():
    # Create dataset
    train_dataset = RellisDataset(path=RELLIS_PATH, split='train')

    for i in range(len(train_dataset)):
        # Get sample
        img, mask = train_dataset[i]
        img = tensor_to_image(img, cfg['dataset_mean'], cfg['dataset_std'])

        if 5 in np.unique(mask):
            mask = mask_to_color(mask, cfg['priseg_dataset'])
            # Visualize sample
            visualize(image=img, mask=mask)


if __name__ == "__main__":
    rellis_demo()
