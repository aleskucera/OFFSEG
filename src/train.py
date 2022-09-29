import os
import logging

import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
import albumentations as A
from omegaconf import DictConfig
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from .utils import mIoU, pixel_accuracy, create_dataset, \
    History, State, get_lr, save_model


def train_model(cfg: DictConfig) -> History:
    log = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # Transformations
    t_train = A.Compose([A.Resize(320, 512, interpolation=cv2.INTER_NEAREST),
                         A.HorizontalFlip(), A.VerticalFlip(),
                         A.GridDistortion(p=0.2),
                         A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
                         A.GaussNoise()])

    t_val = A.Compose([A.Resize(320, 512, interpolation=cv2.INTER_NEAREST),
                       A.HorizontalFlip(),
                       A.GridDistortion(p=0.2)])

    # Datasets
    train_set = create_dataset(cfg, 'train', t_train)
    val_set = create_dataset(cfg, 'val', t_val)

    # Loaders
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=os.cpu_count() // 2)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size,
                            shuffle=False, num_workers=os.cpu_count() // 2)

    # model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=5, activation=None, encoder_depth=5,
    #                  decoder_channels=[256, 128, 64, 32, 16])

    model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=6,  # model output channels (number of classes in your dataset)
    )
    model.to(device)

    history = History()
    state = State(train_num_batches=len(train_loader), val_num_batches=len(val_loader))

    # Initialize criterion, optimizer and scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.train.ignore_index)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.train.learning_rate,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    cfg.train.learning_rate,
                                                    epochs=cfg.train.n_epochs,
                                                    steps_per_epoch=state.train_num_batches)

    for e in range(cfg.train.n_epochs):
        # training loop
        model.train()
        state.reset_train_state()
        for i, data in enumerate(tqdm(train_loader)):
            image_batch, mask_batch = data
            image_batch = image_batch.to(device)
            mask_batch = mask_batch.to(device)

            # Forward pass
            output = model(image_batch)
            loss = criterion(output, mask_batch)

            # evaluation metrics
            state.train_iou_score += mIoU(output, mask_batch)
            state.train_accuracy += pixel_accuracy(output, mask_batch)

            # compute gradient and make an optimization step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # step the learning rate
            history.lrs.append(get_lr(optimizer))
            scheduler.step()

            state.train_loss += loss.item()

        else:
            # validation loop
            model.eval()
            state.reset_val_state()
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    image_batch, mask_batch = data
                    image_batch = image_batch.to(device)
                    mask_batch = mask_batch.to(device)

                    # Forward pass
                    output = model(image_batch)

                    # evaluation metrics
                    state.val_iou_score += mIoU(output, mask_batch)
                    state.val_accuracy += pixel_accuracy(output, mask_batch)

                    # loss
                    loss = criterion(output, mask_batch)
                    state.val_loss += loss.item()

            # average loss and metrics by number of batches
            state.average_metrics()

            # save state to history
            history.save_state(state)

            if state.min_loss_exceeded():
                log.info(f'Loss Decreasing... {state.min_loss:.3f} >> {state.val_loss:.3f}')
                state.update_min_loss()
                if history.num_decreases() % 3 == 0:
                    log.info('Saving model...')
                    save_model(model, cfg, state)

            if history.epochs_stagnated() > cfg.train.patience:
                log.info('Early stopping...')
                break

    save_model(model, cfg, state)
    return history
