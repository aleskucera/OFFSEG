#!/usr/bin/env python
import os
import sys
import test
import train
import logging

# add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

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
              "dataset_size": None,
              "img_size": (320, 512),
              "n_workers": os.cpu_count(),
              "architecture": "fcn_resnet50"},
    "test": {"img_size": (320, 512),
             "model_name": "fcn_resnet50_lr_1e-05_bs_1_epoch_16_ds_10_iou_0.98.pth"}
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
        test.test_model(parameter_parser)


if __name__ == '__main__':
    main()
    logger.info("Done")
