import os

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .classes import State


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_model(model: nn.Module, cfg: DictConfig, state: State):
    file_name = f'{cfg.model.name}_iou-{state.val_iou_score:.3f}_acc-{state.val_accuracy:.3f}.pt'
    file_path = os.path.join(cfg.path.models, file_name)
    torch.save(model, file_path)
