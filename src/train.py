#!/usr/bin/env python

import os
import time
import torch
import logging
import numpy as np
import torchvision.models.segmentation

from utils import IoU
from tqdm import tqdm
from utils import DiceLoss
from dataset import OFFSEG
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from parameter_parser import ParametersImage


def create_model(architecture, n_inputs, n_outputs, pretrained=True):
    """
    Creates a model based on the architecture.

    :param architecture: Architecture of the model
    :param n_inputs: Number of input channels
    :param n_outputs: Number of output channels
    :param pretrained: Use pretrained weights
    """

    # Create logger
    logger = logging.getLogger(__name__)
    assert architecture in ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101',
                            'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large']

    logger.info(f'Creating model {architecture} with {n_inputs} inputs and {n_outputs} outputs')
    Architecture = eval(f'torchvision.models.segmentation.{architecture}')
    model = Architecture(pretrained=pretrained)

    arch = architecture.split('_')[0]
    encoder = '_'.join(architecture.split('_')[1:])

    # Change input layer to accept n_inputs
    if encoder == 'mobilenet_v3_large':
        model.backbone['0'][0] = torch.nn.Conv2d(n_inputs, 16,
                                                 kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    else:
        model.backbone['conv1'] = torch.nn.Conv2d(n_inputs, 64,
                                                  kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Change final layer to output n classes
    if arch == 'lraspp':
        model.classifier.low_classifier = torch.nn.Conv2d(40, n_outputs, kernel_size=(1, 1), stride=(1, 1))
        model.classifier.high_classifier = torch.nn.Conv2d(128, n_outputs, kernel_size=(1, 1), stride=(1, 1))
    elif arch == 'fcn':
        model.classifier[-1] = torch.nn.Conv2d(512, n_outputs, kernel_size=(1, 1), stride=(1, 1))
    elif arch == 'deeplabv3':
        model.classifier[-1] = torch.nn.Conv2d(256, n_outputs, kernel_size=(1, 1), stride=(1, 1))

    return model


def train_model(p: ParametersImage) -> None:
    """
    Train the model, save the weights of the model and save plot of the training process.

    :param p: ParametersImage object where are stored all the parameters
    :return: None
    """

    # Create logger
    logger = logging.getLogger(__name__)

    # Create train dataset
    train_dataset = OFFSEG(path=p.data_path, split='train', crop_size=p.img_size, size=p.dataset_size)
    train_dataloader = DataLoader(train_dataset, batch_size=p.batch_size, shuffle=True, num_workers=p.n_workers // 2)

    # Create validation dataset
    val_dataset = OFFSEG(path=p.data_path, split='val', crop_size=p.img_size, size=p.dataset_size)
    val_dataloader = DataLoader(val_dataset, batch_size=p.batch_size, shuffle=False, num_workers=p.n_workers // 2)

    # Create model
    n_inputs = train_dataset[0][0].shape[0]
    n_outputs = len(train_dataset.class_values)
    model = create_model(p.architecture, n_inputs, n_outputs, pretrained=False)
    model = model.to(p.device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=p.lr)

    # Create loss function
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    criterion = DiceLoss(mode='multilabel', from_logits=True, ignore_index=0)

    # Create IoU metric
    iou = IoU(threshold=0.5, activation='softmax2d', ignore_channels=[0])

    # calculate steps per epoch for training and test set
    train_steps = len(train_dataset) // p.batch_size
    test_steps = len(val_dataset) // p.batch_size

    # initialize a dictionary to store training history
    data = {"train_loss": [], "metric": [], "time": None}

    # Create progress bar
    pbar = tqdm(total=(len(train_dataset) + len(val_dataset)) * p.n_epochs)

    max_avg_metric = -np.Inf
    start_time = time.time()
    for epoch in range(p.n_epochs):

        logger.info(f"Entering epoch {epoch + 1} of {p.n_epochs} epochs")

        # initialize the total training and validation loss for the current epoch
        total_train_loss = 0
        total_val_metric = 0

        # -------- TRAINING PHASE --------
        model = model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            # Move images and labels to device
            images = images.to(p.device)
            labels = labels.to(p.device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs['out'], labels.long())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add the loss to the total training loss so far
            total_train_loss += loss

            # Update progress bar
            pbar.set_description(f'Epoch: {epoch}, Phase: training, Loss {loss:.4f}')
            pbar.update(1)

        # -------- VALIDATION PHASE --------
        model = model.eval()
        for i, (images, labels) in enumerate(val_dataloader):
            # Move images and labels to device
            images = images.to(p.device)
            labels = labels.to(p.device)

            # convert tensor to int values
            with torch.no_grad():
                outputs = model(images)
                metric = iou(outputs['out'], labels.long())
                total_val_metric += metric

            # Update progress bar
            pbar.set_description(f'Epoch: {epoch}, Phase: validation, IoU {metric:.2f}')
            pbar.update(1)

        avg_val_metric = total_val_metric / test_steps
        avg_train_loss = total_train_loss / train_steps

        logger.debug(f"Average training loss: {avg_train_loss:.4f}")
        logger.debug(f"Average validation IoU: {avg_val_metric:.4f}")

        # save the model if it is the best so far
        if max_avg_metric < avg_val_metric:
            max_avg_metric = avg_val_metric
            name = f'{p.architecture}_lr_{p.lr}' \
                   f'_bs_{p.batch_size}_epoch_{epoch}' \
                   f'_ds_{p.dataset_size}_iou_{max_avg_metric:.2f}.pth'
            logger.info(f"Saving Model: {name}")
            os.makedirs(p.save_path, exist_ok=True)
            torch.save(model, os.path.join(p.save_path, name))

        # update our training history
        data["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        data["metric"].append(avg_val_metric.cpu().detach().numpy())

    # Close progress bar
    pbar.close()

    # display the total time needed to perform the training
    end_time = time.time()
    data["time"] = end_time - start_time
    logger.info(f"Training completed in {data['time']:.2f} seconds")
    plot_training_history(p, data)


def plot_training_history(p: ParametersImage, data: dict, optimizer: str = 'Adam', loss: str = 'DiceLoss',
                          metric: str = 'IoU'):
    """
    Creates and saves a plot of the training history of the model.
    The name of the plot is in following format:
        {architecture}
        _lr_{learning_rate}
        _bs_{batch_size}
        _ds_{dataset_size}
        _is_{images_size}
        _opt_{optimizer}
        _l_{learning_rate}
        _m_{metric}.png

    :param p: ParametersImage
    :param data: dict
    :param optimizer: str
    :param loss: str
    :param metric: str
    :return: None
    """

    # Create name of the plot
    name = f'{p.architecture}' \
           f'_lr_{p.lr}' \
           f'_ds_{p.dataset_size}' \
           f'_is_{p.img_size}' \
           f'_opt_{optimizer}' \
           f'_l_{loss}' \
           f'_m_{metric}.png'
    plot_dir = os.path.join(os.path.dirname(__file__), "..", "log", "figures")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, name)

    # Create logger
    logger = logging.getLogger(__name__)
    logger.info(f"Plotting training history and saving to {plot_path}")

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, len(data["train_loss"])), data["train_loss"], label="train_loss")
    plt.plot(np.arange(0, len(data["metric"])), data["metric"], label="metric")
    plt.title("Training Loss and Metric")
    plt.xlabel("Epoch [-]")
    plt.ylabel("Loss/Metric [-]")
    plt.legend(loc="best")
    plt.savefig(plot_path)
    # plt.show()
