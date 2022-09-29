import os
from .base_dataset import BaseDataset


class RellisDataset(BaseDataset):
    def __init__(self, path, split=None, size=None, transform=None, mean=None, std=None, color_map=None):
        self.ds_str = {'image_dir': 'pylon_camera_node',
                       'label_dir': 'pylon_camera_node_label_id',
                       'image_ext': '.jpg',
                       'label_ext': '.png'}

        super().__init__(path, 'rellis', split, size, transform, mean=mean, std=std, color_map=color_map)
        self.images, self.labels = self._read_files()
        self.generate_split()

    def _read_files(self):
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
