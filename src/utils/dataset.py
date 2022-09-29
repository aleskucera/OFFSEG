import torch
from tqdm import tqdm
from torch.utils.data import ConcatDataset

from src.dataset import RellisDataset, RugdDataset, CityscapesDataset


def calculate_mean_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for image, _ in tqdm(dataloader):
        channels_sum += torch.mean(image, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(image ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


def create_dataset(cfg, split, transform):
    rellis_dataset = RellisDataset(path=cfg.path.rellis, split=split, mean=cfg.ds.mean, std=cfg.ds.std,
                                   color_map=cfg.rellis, transform=transform, size=cfg.train.dataset_size)
    rugd_dataset = RugdDataset(path=cfg.path.rugd, split=split, mean=cfg.ds.mean, std=cfg.ds.std, color_map=cfg.rugd,
                               transform=transform, size=cfg.train.dataset_size)
    cityscapes_dataset = CityscapesDataset(path=cfg.path.cityscapes, split=split, mean=cfg.ds.mean, std=cfg.ds.std,
                                           color_map=cfg.cityscapes, transform=transform, size=cfg.train.dataset_size)
    return ConcatDataset([rellis_dataset, rugd_dataset, cityscapes_dataset])
