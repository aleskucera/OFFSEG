#!/usr/bin/env python
import os
import sys
import test
import train
import logging
import matplotlib.pyplot as plt

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
FIGURE_PATH = os.path.join(os.path.dirname(__file__), "..", "log/figures")
PARAMETERS = {
    "train": {"lr": 1e-3,
              "n_epochs": 10,
              "batch_size": 1,
              "dataset_size": 100,
              "img_size": (320, 512),
              "n_workers": os.cpu_count(),
              "architecture": "fcn_resnet50"},
    "test": {"img_size": (320, 512),
             "model_name": "fcn_resnet50_lr_1e-05_bs_1_epoch_16_ds_10_iou_0.98.pth"}
}


def plot_loss(history):
    plt.figure()
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.plot(history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig(os.path.join(FIGURE_PATH, 'loss.png'))


def plot_score(history):
    plt.figure()
    plt.plot(history['train_mIoU'], label='train_mIoU', marker='*')
    plt.plot(history['val_mIoU'], label='val_mIoU', marker='*')
    plt.title('Score per epoch')
    plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig(os.path.join(FIGURE_PATH, 'plot.png'))


def plot_acc(history):
    plt.figure()
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['val_acc'], label='val_accuracy', marker='*')
    plt.title('Accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig(os.path.join(FIGURE_PATH, 'acc.png'))


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
        history = train.train_model(parameter_parser)
        plot_loss(history)
        plot_score(history)
        plot_acc(history)
    elif parameter_parser.mode == "test":
        test.test_model()


if __name__ == '__main__':
    main()
    logger.info("Done")
