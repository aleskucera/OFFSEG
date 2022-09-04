import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


def convert_color(label, color_map):
    if isinstance(label, np.ndarray):
        temp = np.zeros(label.shape + (3,)).astype(np.uint8)
    elif isinstance(label, torch.Tensor):
        temp = torch.zeros(label.shape + (3,), dtype=torch.uint8)
    else:
        raise ValueError('Supported types: np.ndarray, torch.Tensor')
    for k, v in color_map.items():
        temp[label == k] = v
    return temp


def convert_label(label, inverse=False, label_mapping=None):
    assert label_mapping is not None
    if isinstance(label, np.ndarray):
        temp = label.copy()
    elif isinstance(label, torch.Tensor):
        temp = label.clone()
    else:
        raise ValueError('Supported types: np.ndarray, torch.Tensor')
    if inverse:
        for v, k in label_mapping.items():
            temp[label == k] = v
    else:
        for k, v in label_mapping.items():
            temp[label == k] = v
    return temp


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    @staticmethod
    def forward(inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = create_one_hot(inputs)
        inputs = inputs.flatten(2, 3)
        targets = targets.flatten(2, 3)
        intersection = torch.sum(inputs * targets, dim=2)
        inputs_sum = torch.sum(inputs, dim=2)
        targets_sum = torch.sum(targets, dim=2)
        dice = torch.mean((2. * intersection + smooth) / (inputs_sum + targets_sum + smooth))

        return 1 - dice


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    @staticmethod
    def forward(inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = create_one_hot(inputs)
        inputs = inputs.flatten(2, 3)
        targets = targets.flatten(2, 3)
        intersection = torch.sum(inputs * targets, dim=2)
        total = torch.sum(inputs + targets, dim=2)
        union = total - intersection
        IoU = torch.mean((intersection + smooth) / (union + smooth))
        return IoU


def create_one_hot(inputs):
    print(inputs.shape)
    inputs = torch.argmax(inputs, dim=1)
    one_hot = F.one_hot(inputs, num_classes=5)
    one_hot = one_hot.permute(0, 3, 1, 2)
    print(one_hot.shape)
    return one_hot


def main():
    inputs = torch.rand(1, 5, 2, 2)
    print(inputs)
    print(inputs.log_softmax(dim=1).exp())
    # test dice loss
    dice_loss = DiceLoss()
    inputs = torch.randint(low=0, high=2, size=(1, 5, 2, 2))
    targets = torch.randint(low=0, high=2, size=(1, 5, 2, 2))
    print(dice_loss(inputs, targets))

    # test IoU loss
    iou_loss = IoULoss()
    inputs = torch.rand(1, 5, 256, 256)
    targets = torch.rand(1, 5, 256, 256)
    print(iou_loss(inputs, targets))


if __name__ == '__main__':
    main()
