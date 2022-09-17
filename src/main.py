#!/usr/bin/env python
import os
import sys
import test
import train
import logging
import argparse

# add the root directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from log import configure_logging
from src.utils import plot_loss, plot_score, plot_acc

# Create logger
CLEAR_LOGS = True  # Clear logs before running
configure_logging(clear_logs=CLEAR_LOGS)
logger = logging.getLogger(__name__)


def main():
    # Create the config object
    cfg = Config()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--action', type=str, default='train', help='Action to perform: train or test')
    parser.add_argument('--dev', action='store_true',
                        help='Optional argument if you want to run in development mode')
    args = parser.parse_args()

    # Train or test the model
    if args.action == "train":
        if args.dev:
            parameters = cfg['train_params_dev']
        else:
            parameters = cfg['train_params']
        history = train.train_model(p=parameters, save_path=cfg["models_dir"])
        plot_loss(history, save_path=cfg["plots_dir"])
        plot_score(history, save_path=cfg["plots_dir"])
        plot_acc(history, save_path=cfg["plots_dir"])
    elif args.action == "test":
        test.test_model()


if __name__ == '__main__':
    main()
    logger.info("Done")
