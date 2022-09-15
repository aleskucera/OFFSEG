#!/usr/bin/env python

import os
import cv2
import torch
import logging
import numpy as np

from dataset import OFFSEG
from dataset import visualize, convert_color
from dataset import TRAVERSABILITY_COLOR_MAP
from torch.utils.data import DataLoader
from parameter_parser import ParametersImage


def test_model(p: ParametersImage) -> None:
    """
    Test the model and visualize the predictions

    :param p: Parameters object
    :return: None
    """

    # Create logger
    logger = logging.getLogger(__name__)

    # Create test dataset
    test_dataset = OFFSEG(path=p.data_path, split='test', crop_size=p.img_size)

    # Load model
    # model = torch.load(os.path.join(os.path.dirname(__file__), "..", "models", p.model_name))
    model = torch.load('Unet-Mobilenet2.pt')
    model.eval()

    for i in range(5):
        # Apply inference preprocessing transforms
        image, gt_mask = test_dataset[i]
        img_vis = np.uint8(255 * (image * test_dataset.std + test_dataset.mean))
        if test_dataset.split == 'test':
            image = image.transpose((2, 0, 1))  # (H x W x C) -> (C x H x W)
        batch = torch.from_numpy(image).unsqueeze(0).to(p.device)

        # Use the model and visualize the prediction
        with torch.no_grad():
            output = model(batch)
        pred = torch.softmax(output['out'], dim=1)
        pred = pred.squeeze(0).cpu().numpy()

        # Find the class with the highest probability
        mask = np.argmax(pred, axis=0)
        gt_mask = np.argmax(gt_mask, axis=0)

        # Resize the mask to the image size
        size = (p.img_size[1], p.img_size[0])
        mask = cv2.resize(mask.astype('float32'), size, interpolation=cv2.INTER_LINEAR).astype('int8')

        # result = convert_color(mask, data_cfg['color_map'])
        result = convert_color(mask, TRAVERSABILITY_COLOR_MAP)
        gt_result = convert_color(gt_mask, TRAVERSABILITY_COLOR_MAP)

        # Visualize the prediction
        visualize(img=img_vis, label=result, gt_label=gt_result)
