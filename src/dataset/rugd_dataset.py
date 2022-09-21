import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from config import Config
from .base_dataset import BaseDataset
from src.utils import visualize, tensor_to_image, mask_to_color

cfg = Config()

RUGD_PATH = '/home/ales/Datasets/RUGD'


class RugdDataset(BaseDataset):
    def __init__(self, path, split=None, size=None, transform=None):
        self.ds_str = {'image_dir': 'RUGD_frames-with-annotations',
                       'label_dir': 'RUGD_annotations',
                       'ext': '.png'}

        super().__init__(path, 'rugd_dataset', split, size, transform)
        self.images, self.labels = self.read_files()
        self.generate_split()

    def read_files(self):
        images, labels = [], []

        # Walk through the dataset path until we find label, then find the corresponding image
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if self.ds_str['label_dir'] in root and file.endswith(self.ds_str['ext']):
                    # Create paths
                    label_path = os.path.join(root, file)
                    image_path = label_path.replace(self.ds_str['label_dir'], self.ds_str['image_dir'])

                    # Append paths
                    labels.append(label_path)
                    images.append(image_path)

        # Resize dataset for development purposes (1/3 of the size because of 3 datasets)
        if self.size is not None:
            images = images[:self.size // 3]
            labels = labels[:self.size // 3]
        return images, labels


def rugd_demo():
    # Create dataset
    dataset = RugdDataset(path=RUGD_PATH, split='train', size=100)
    print(len(dataset))

    for i in range(len(dataset)):
        # Get sample
        img, mask = dataset[i]

        # Prepare image and label for visualization
        img = tensor_to_image(img, cfg['dataset_mean'], cfg['dataset_std'])
        mask = mask_to_color(mask, cfg['priseg_dataset'])

        # Visualize
        visualize(image=img, mask=mask)


if __name__ == "__main__":
    rugd_demo()
