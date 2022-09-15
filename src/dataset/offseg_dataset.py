import os
import cv2
import torch
import numpy as np
import logging
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '../..'))


class OFFSEGDataset(Dataset):
    def __init__(self, split=None, size=None,
                 path=None, transform=None,
                 mean=(0.44833934, 0.49257269, 0.46350682),
                 std=(0.22696872, 0.23755784, 0.27277329)):

        # Check arguments
        if path is None:
            path = os.path.join(root_dir, 'data')
        assert os.path.exists(path)
        assert split in [None, 'train', 'val', 'test']

        # Set attributes
        self.std = std
        self.mean = mean
        self.path = path
        self.size = size
        self.split = split
        self.transform = transform

        # Read files
        self.images, self.labels = self.read_files()
        self.generate_split()
        self.info()

    def info(self):
        logger = logging.getLogger(__name__)
        logger.info(f"Generated {self.split} split of size {self.__len__()}")
        logger.info(f"Normalized dataset with the mean {self.mean} and std {self.std}")

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

    def show_samples(self):

        # Show 3 random samples
        for i in range(3):
            idx = np.random.randint(0, len(self.images))
            img = cv2.imread(self.images[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.labels[idx], cv2.IMREAD_GRAYSCALE)
            plt.figure(figsize=(20, 10))
            plt.title('Image with ground truth')
            plt.subplot(1, 2, 1)
            plt.axis('off')
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.axis('off')
            plt.imshow(img)
            plt.imshow(mask, alpha=0.6)
            plt.show()

    # calculate mean and std of the dataset
    def calculate_mean_std(self):
        mean = [0, 0, 0]
        std = [0, 0, 0]
        for i in range(len(self.images)):
            print(f"\rCalculating mean and std for image {i + 1}/{len(self.images)}", end='')
            img = cv2.imread(self.images[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            mean += np.mean(img, axis=(0, 1))
            std += np.std(img, axis=(0, 1))
        mean /= len(self.images)
        std /= len(self.images)
        return mean, std


if __name__ == "__main__":
    dataset = OFFSEGDataset(split='train', size=1000)
    dataset.show_samples()
