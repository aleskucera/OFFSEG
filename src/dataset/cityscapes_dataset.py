import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from config import Config
from .base_dataset import BaseDataset
from src.utils import visualize, tensor_to_image, mask_to_color

cfg = Config()

CITYSCAPES_PATH = '/home/ales/Datasets/cityscapes'


class CityscapesDataset(BaseDataset):
    def __init__(self, path, split=None, size=None, transform=None):
        self.ds_str = {'image_dir': 'leftImg8bit',
                       'label_dir': 'gtFine',
                       'image_end': 'leftImg8bit',
                       'label_end': 'gtFine_labelIds',
                       'ext': '.png'}

        super().__init__(path, 'cityscapes_dataset', split, size, transform)

        self.images, self.labels = self.read_files()

    def read_files(self):
        images, labels = [], []
        images_path = os.path.join(self.path, self.ds_str['image_dir'], self.split)

        # Walk through the images path until we find an image, then find the corresponding label
        for root, dirs, files in os.walk(images_path):
            for file in files:
                if file.endswith(self.ds_str['ext']):
                    # Create paths
                    image_path = os.path.join(root, file)
                    label_path_split = image_path.split('/')
                    label_path_split[-4] = self.ds_str['label_dir']
                    label_path_split[-1] = label_path_split[-1].replace(self.ds_str['image_end'],
                                                                        self.ds_str['label_end'])
                    label_path = '/'.join(label_path_split)

                    # Append paths
                    labels.append(label_path)
                    images.append(image_path)

        # Resize dataset for development purposes (1/3 of the size because of 3 datasets)
        if self.size is not None:
            images = images[:self.size // 3]
            labels = labels[:self.size // 3]
        return images, labels


def cityscapes_demo():
    # Create dataset
    dataset = CityscapesDataset(path=CITYSCAPES_PATH, split='train')
    print(len(dataset))

    for i in range(5):
        # Get sample
        img, mask = dataset[i]
        img = tensor_to_image(img, cfg['dataset_mean'], cfg['dataset_std'])

        mask = mask_to_color(mask, cfg['priseg_dataset'])
        # Visualize sample
        visualize(image=img, mask=mask)


if __name__ == "__main__":
    # cityscapes_demo()

    arr = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    print(arr)
    print(np.sum(arr, axis=(0, 1)))
