import os
import time
import logging

import cv2
import torch
import numpy as np
import torch.nn as nn
import albumentations as A
import segmentation_models_pytorch as smp

from tqdm import tqdm
from dataset import RellisDataset, RugdDataset, CityscapesDataset
from torch.utils.data import DataLoader, ConcatDataset
from utils import mIoU, pixel_accuracy


def create_dataset(rellis_path, rugd_path, cityscapes_path, split, size=None, transform=None):
    rellis_dataset = RellisDataset(path=rellis_path, split=split, size=size, transform=transform)
    rugd_dataset = RugdDataset(path=rugd_path, split=split, size=size, transform=transform)
    cityscapes_dataset = CityscapesDataset(path=cityscapes_path, split=split, size=size, transform=transform)
    return ConcatDataset([rellis_dataset, rugd_dataset, cityscapes_dataset])


def calculate_mean_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for image, _ in tqdm(dataloader):
        channels_sum += torch.mean(image, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(image ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


def train_model(p: dict, save_path: str) -> dict:
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
    train_set = create_dataset(p['rellis_path'], p['rugd_path'], p['cityscapes_path'],
                               'train', size=p['dataset_size'], transform=t_train)
    val_set = create_dataset(p['rellis_path'], p['rugd_path'], p['cityscapes_path'],
                             'val', size=p['dataset_size'], transform=t_val)

    # Loaders
    train_loader = DataLoader(train_set, batch_size=p['batch_size'], shuffle=True, num_workers=os.cpu_count() // 2)
    val_loader = DataLoader(val_set, batch_size=p['batch_size'], shuffle=False, num_workers=os.cpu_count() // 2)
    # model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=5, activation=None, encoder_depth=5,
    #                  decoder_channels=[256, 128, 64, 32, 16])

    model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=6,  # model output channels (number of classes in your dataset)
    )

    model.to(device)
    torch.cuda.empty_cache()

    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': [],
               'train_mIoU': [], 'val_mIoU': [],
               'lrs': []}

    min_loss = np.inf
    decrease = 1
    not_improve = 0

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, p['learning_rate'], epochs=p['n_epochs'],
                                                    steps_per_epoch=len(train_loader))

    fit_time = time.time()
    for e in range(p['n_epochs']):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image_batch, mask_batch = data
            image_batch = image_batch.to(device)
            mask_batch = mask_batch.to(device)

            # Forward pass
            output = model(image_batch)
            loss = criterion(output, mask_batch)

            # evaluation metrics
            iou_score += mIoU(output, mask_batch)
            accuracy += pixel_accuracy(output, mask_batch)

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
            val_loss = 0
            val_accuracy = 0
            val_iou_score = 0
            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    # reshape to 9 patches from single image, delete batch size
                    image_batch, mask_batch = data
                    image_batch = image_batch.to(device)
                    mask_batch = mask_batch.to(device)

                    # Forward pass
                    output = model(image_batch)

                    # evaluation metrics
                    val_iou_score += mIoU(output, mask_batch)
                    val_accuracy += pixel_accuracy(output, mask_batch)

                    # loss
                    loss = criterion(output, mask_batch)
                    val_loss += loss.item()

            # save history
            history['train_loss'].append(running_loss / len(train_loader))
            history['val_loss'].append(val_loss / len(val_loader))

            if min_loss > (val_loss / len(val_loader)):
                logger.info(f'Loss Decreasing.. {min_loss:.3f} >> {val_loss / len(val_loader):.3f}')
                min_loss = (val_loss / len(val_loader))
                decrease += 1
                if decrease % 5 == 0:
                    logger.info('Saving model...')
                    torch.save(model, os.path.join(save_path,
                                                   f'Resnet34-{val_iou_score / len(val_loader):.3f}.pt'))

            if (val_loss / len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = (val_loss / len(val_loader))
                logger.info(f"Loss didn't decreased for {not_improve} time")
                if not_improve == 7:
                    print("Loss didn't decreased for 7 times, Stop Training")
                    break

            history['train_acc'].append(accuracy / len(train_loader))
            history['train_mIoU'].append(iou_score / len(train_loader))
            history['val_acc'].append(val_accuracy / len(val_loader))
            history['val_mIoU'].append(val_iou_score / len(val_loader))

            logger.info(f"\nEpoch: {e + 1}/{p['n_epochs']} \n"
                        f"Train Loss: {running_loss / len(train_loader):.2f} \n"
                        f"Val Loss: {val_loss / len(val_loader):.2f} \n"
                        f"Train Accuracy: {accuracy / len(train_loader):.2f} \n"
                        f"Val Accuracy: {val_accuracy / len(val_loader):.2f} \n"
                        f"Train mIoU: {iou_score / len(train_loader):.2f} \n"
                        f"Val mIoU: {val_iou_score / len(val_loader):.2f} \n"
                        f"Time: {time.time() - since:.2f} \n")

    logger.info(f'Total time: {(time.time() - fit_time) / 60:.2f} m')
    torch.save(model, os.path.join(save_path, f'Resnet34.pt'))
    return history


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
