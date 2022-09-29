import os

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.dataset import CityscapesDataset, RellisDataset, RugdDataset
from src.utils import visualize, tensor_to_image, mask_to_color

DATASET = 'rugd'


@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def dataset_demo(cfg: DictConfig) -> None:
    # Create dataset
    if DATASET == 'cityscapes':
        dataset = CityscapesDataset(path=cfg.path.cityscapes, split='train', mean=cfg.ds.mean,
                                    std=cfg.ds.std, color_map=cfg.cityscapes)
    elif DATASET == 'rellis':
        dataset = RellisDataset(path=cfg.path.rellis, split='train', mean=cfg.ds.mean,
                                std=cfg.ds.std, color_map=cfg.rellis)
    elif DATASET == 'rugd':
        dataset = RugdDataset(path=cfg.path.rugd, split='train', mean=cfg.ds.mean,
                              std=cfg.ds.std, color_map=cfg.rugd)
    else:
        raise ValueError('Dataset not supported')

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=os.cpu_count() // 2)

    # Show samples
    show_samples(loader, cfg.ds.mean, cfg.ds.std, cfg.ds.color_map)


def show_samples(loader, mean, std, color_map, num_samples=5):
    for i, (img, mask) in enumerate(loader):
        img = tensor_to_image(img.squeeze(), mean, std)
        mask = mask_to_color(mask.squeeze(), color_map)
        visualize(image=img, mask=mask)
        if i == num_samples:
            break


def show_label_examples(loader, label, mean, std, color_map, num_samples=5):
    for (img, mask) in loader:
        if label in mask:
            img = tensor_to_image(img, mean, std)
            mask = mask_to_color(mask, color_map)
            visualize(image=img, mask=mask)
            num_samples -= 1
            if num_samples == 0:
                break


if __name__ == "__main__":
    dataset_demo()
