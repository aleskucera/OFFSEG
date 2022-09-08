import torch
import numpy as np
import matplotlib.pyplot as plt


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
