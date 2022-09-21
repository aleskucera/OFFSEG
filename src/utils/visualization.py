import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T


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


def convert_color(label, color_map):
    if isinstance(label, np.ndarray):
        temp = np.zeros(label.shape + (3,)).astype(np.uint8)
        for k, v in color_map.items():
            temp[label == k] = v
    elif isinstance(label, torch.Tensor):
        temp = torch.zeros(label.shape + (3,), dtype=torch.uint8)
        for k, v in color_map.items():
            temp[label == k] = torch.ByteTensor(v)
    else:
        raise ValueError('Supported types: np.ndarray, torch.Tensor')
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


def mask_to_color(mask, dataset_config):
    # Map label to color
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for v in dataset_config.values():
        rgb_mask[mask == v['id']] = v['color']
    return rgb_mask


def color_to_mask(color_mask, dataset_config):
    mask = np.zeros((color_mask.shape[0], color_mask.shape[1]), dtype=np.uint8)
    for v in dataset_config.values():
        mask[np.where(np.all(color_mask == v['color'], axis=-1))] = v['id']
    return mask


def tensor_to_image(tensor: torch.Tensor, mean: list, std: list):
    inv_std = [1 / s for s in std]
    zero_mean = [0 for _ in mean]
    neg_mean = [-m for m in mean]
    t = T.Compose([T.Normalize(zero_mean, inv_std),
                   T.Normalize(neg_mean, [1, 1, 1]),
                   T.ToPILImage()])
    return t(tensor)
