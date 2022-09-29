import os
from .base_dataset import BaseDataset


class CityscapesDataset(BaseDataset):
    def __init__(self, path, split=None, size=None, transform=None, mean=None, std=None, color_map=None):
        self.ds_str = {'image_dir': 'leftImg8bit',
                       'label_dir': 'gtFine',
                       'image_end': 'leftImg8bit',
                       'label_end': 'gtFine_labelIds',
                       'ext': '.png'}

        super().__init__(path, 'cityscapes', split, size, transform, mean=mean, std=std, color_map=color_map)

        self.images, self.labels = self._read_files()

    def _read_files(self):
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
