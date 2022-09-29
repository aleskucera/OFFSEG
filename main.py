#!/usr/bin/env python
import logging

import hydra
from omegaconf import DictConfig

from src import train_model, test_model, plot_results

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(f"Dataset size: {cfg.train.dataset_size}")
    if cfg.action == "train":
        history = train_model(cfg)
        plot_results(history, save_path=cfg.path.plots)
    elif cfg.action == "test":
        test_model(cfg)


if __name__ == '__main__':
    main()
