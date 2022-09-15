import os
import cv2
import time
import torch
import logging
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import albumentations as A
import matplotlib.pyplot as plt
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from PIL import Image
from tqdm import tqdm
from dataset import OFFSEGDataset
from torchsummary import summary
from torch.autograd import Variable
from torchvision import transforms as T
from parameter_parser import ParametersImage
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def train_model(p: ParametersImage) -> dict:
    # Create logger
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations
    t_train = A.Compose([A.Resize(320, 512, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.VerticalFlip(),
                         A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
                         A.GaussNoise()])

    t_val = A.Compose([A.Resize(320, 512, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(),
                       A.GridDistortion(p=0.2)])

    # Datasets
    train_set = OFFSEGDataset('train', size=p.dataset_size, transform=t_train)
    val_set = OFFSEGDataset('val', size=p.dataset_size, transform=t_val)

    # Loaders
    train_loader = DataLoader(train_set, batch_size=p.batch_size, shuffle=True, num_workers=p.n_workers // 2)
    val_loader = DataLoader(val_set, batch_size=p.batch_size, shuffle=False, num_workers=p.n_workers // 2)

    model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=5, activation=None, encoder_depth=5,
                     decoder_channels=[256, 128, 64, 32, 16])
    model.to(device)
    torch.cuda.empty_cache()

    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': [],
               'train_mIoU': [], 'val_mIoU': [],
               'lrs': []}

    min_loss = np.inf
    decrease = 1
    not_improve = 0

    weight_decay = 1e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=p.lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, p.lr, epochs=p.n_epochs,
                                                    steps_per_epoch=len(train_loader))

    fit_time = time.time()
    for e in range(p.n_epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = data
            image = image_tiles.to(device)
            mask = mask_tiles.to(device)

            # Forward pass
            output = model(image)
            loss = criterion(output, mask)
            # evaluation metrics
            iou_score += mIoU(output, mask)
            accuracy += pixel_accuracy(output, mask)
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            history['lrs'].append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    # reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)
                    output = model(image)
                    # evaluation metrics
                    val_iou_score += mIoU(output, mask)
                    test_accuracy += pixel_accuracy(output, mask)
                    # loss
                    loss = criterion(output, mask)
                    test_loss += loss.item()

            # save history
            history['train_loss'].append(running_loss / len(train_loader))
            history['val_loss'].append(test_loss / len(val_loader))

            if min_loss > (test_loss / len(val_loader)):
                logger.info(f'Loss Decreasing.. {min_loss:.3f} >> {test_loss / len(val_loader):.3f}')
                min_loss = (test_loss / len(val_loader))
                decrease += 1
                if decrease % 5 == 0:
                    logger.info('Saving model...')
                    torch.save(model, os.path.join(p.save_path,
                                                   f'Unet-Mobilenet_v2_mIoU-{val_iou_score / len(val_loader):.3f}.pt'))

            if (test_loss / len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = (test_loss / len(val_loader))
                logger.info(f"Loss didn't decreased for {not_improve} time")
                if not_improve == 7:
                    print("Loss didn't decreased for 7 times, Stop Training")
                    break

            history['train_acc'].append(accuracy / len(train_loader))
            history['val_acc'].append(test_accuracy / len(val_loader))
            history['train_mIoU'].append(iou_score / len(train_loader))
            history['val_mIoU'].append(val_iou_score / len(val_loader))

            logger.info(f"\nEpoch: {e + 1}/{p.n_epochs} \n"
                        f"Train Loss: {running_loss / len(train_loader):.2f} \n"
                        f"Val Loss: {test_loss / len(val_loader):.2f} \n"
                        f"Train Accuracy: {accuracy / len(train_loader):.2f} \n"
                        f"Val Accuracy: {test_accuracy / len(val_loader):.2f} \n"
                        f"Train mIoU: {iou_score / len(train_loader):.2f} \n"
                        f"Val mIoU: {val_iou_score / len(val_loader):.2f} \n"
                        f"Time: {time.time() - since:.2f} \n")

    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    torch.save(model, os.path.join(p.save_path, f'Unet-Mobilenet2.pt'))
    return history


def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=5):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes):  # loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:  # no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
