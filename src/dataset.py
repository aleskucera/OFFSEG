import os
import cv2
import numpy as np

from utils.dataset_utils import visualize, convert_color

from PIL import Image
from sklearn.model_selection import train_test_split
from base_dataset import TRAVERSABILITY_LABELS, TRAVERSABILITY_COLOR_MAP, BaseDatasetImages, VOID_VALUE

root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))


class OFFSEG(BaseDatasetImages):
    CLASSES = ["void", "traversable", "non-traversable", "obstacle", "sky"]

    def __init__(self,
                 path=None,
                 split=None,
                 num_samples=None,
                 multi_scale=True,
                 flip=True,
                 ignore_label=VOID_VALUE,
                 base_size=2048,
                 crop_size=(1200, 1920),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=np.array([0, 0, 0]),
                 std=np.array([1, 1, 1]),
                 size=None):
        super(OFFSEG, self).__init__(ignore_label, base_size,
                                     crop_size, downsample_rate, scale_factor, mean, std)
        if path is None:
            path = os.path.join(root_dir, 'dataset')
        assert os.path.exists(path)
        assert split in [None, 'train', 'val', 'test']
        self.path = path
        self.split = split
        self.size = size

        self.class_values = np.sort([k for k in TRAVERSABILITY_LABELS.keys()])  # [0, 1, 2, 3, 4]
        self.color_map = TRAVERSABILITY_COLOR_MAP

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1. / downsample_rate

        self.multi_scale = multi_scale
        self.flip = flip
        self.rng = np.random.default_rng(seed=42)

        self.images, self.labels = self.read_files()
        self.generate_split()
        if num_samples:
            self.images = self.images[:num_samples]

    def read_files(self):
        images = []
        labels = []
        for directory in os.listdir(self.path):
            seq_path = os.path.join(self.path, directory)
            if os.path.isdir(seq_path):
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
        images = images[:self.size]
        labels = labels[:self.size]
        return images, labels

    def generate_split(self, test_ratio=0.2, ):
        X_train, X_test, y_train, y_test = train_test_split(self.images, self.labels, test_size=test_ratio,
                                                            random_state=42)
        if self.split == 'train':
            self.images = X_train
            self.labels = y_train
        elif self.split in ['val', 'test']:
            self.images = X_test
            self.labels = y_test

    def __getitem__(self, index):
        image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        mask = np.array(Image.open(self.labels[index]))

        if 'test' in self.split:
            new_h, new_w = self.crop_size
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            image = self.input_transform(image)
        else:
            # add augmentations
            image, mask = self.apply_augmentations(image, mask, self.multi_scale, self.flip)

        # extract certain classes from mask
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=0).astype('float')

        return image.copy(), mask.copy()

    def __len__(self):
        return len(self.images)


def main():
    ds = OFFSEG(split='train')
    print(len(ds))

    for _ in range(5):
        i = np.random.choice(range(len(ds)))
        img, label = ds[i]

        img_vis = img.transpose((1, 2, 0)) * ds.std + ds.mean
        label = label.argmax(axis=0)
        mask = convert_color(label, ds.color_map)

        visualize(img=img_vis, label=mask)


if __name__ == '__main__':
    main()
