#!/usr/bin/env python
import os
import train
import logging

from log import configure_logging
from parameter_parser import ParametersImage

# Create logger
CLEAR_LOGS = True  # Clear logs before running
configure_logging(clear_logs=CLEAR_LOGS)
logger = logging.getLogger(__name__)

# Set the parameters
MODE = "train"
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")
PARAMETERS = {
    "train": {"lr": 1e-5,
              "n_epochs": 30,
              "batch_size": 1,
              "dataset_size": 10,
              "img_size": (320, 512),
              "n_workers": os.cpu_count(),
              "architecture": "fcn_resnet50",
              "plot_path": os.path.join(os.path.dirname(__file__), "..", "log", "images", "train_history.png"), },
    "test": {"lr": 1e-5,
             "img_size": (320, 512)}
}


def main():
    # Check macro
    if MODE not in ["train", "test"]:
        logger.error("MODE must be either 'train' or 'test'")
    if not os.path.exists(DATASET_PATH):
        logger.error(f"DATASET_PATH '{DATASET_PATH}' does not exist")
    if not os.path.exists(MODELS_PATH):
        logger.error(f"MODELS_PATH '{MODELS_PATH}' does not exist")
    if not PARAMETERS:
        logger.error("PARAMETERS must be provided")

    # Read command line and create parameter object
    parameter_parser = ParametersImage(mode=MODE, parameters=PARAMETERS, data_path=DATASET_PATH,
                                       save_path=MODELS_PATH)

    # Train or test the model
    if parameter_parser.mode == "train":
        train.train_model(parameter_parser)
    elif parameter_parser.mode == "test":
        pass


if __name__ == '__main__':
    main()
    logger.info("Done")
