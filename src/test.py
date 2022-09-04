#!/usr/bin/env python
# from __future__ import absolute_import

import os
import cv2
import torch
import numpy as np

from dataset import OFFSEG
from utils import convert_color
from argparse import ArgumentParser
from matplotlib import pyplot as plt


def main():
    parser = ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='TraversabilityImagesFiftyone')
    parser.add_argument('--img_size', nargs='+', default=(192, 320))
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    print(args)

    ds = OFFSEG(crop_size=args.img_size, split='test')

    # Initialize model with the best available weights
    model_name = 'fcn_resnet50_lr_1e-05_bs_1_epoch_0_OFFSEG_iou_0.86.pth'
    # model_name = 'fcn_resnet50_lr_1e-05_bs_1_epoch_3_TraversabilityImages_iou_0.71.pth'
    model_path = os.path.join('..', 'models', model_name)
    model = torch.load(model_path, map_location=args.device).eval()

    for i in range(5):
        # Apply inference preprocessing transforms
        img, gt_mask = ds[i]
        img_vis = np.uint8(255 * (img * ds.std + ds.mean))
        if ds.split == 'test':
            img = img.transpose((2, 0, 1))  # (H x W x C) -> (C x H x W)
        batch = torch.from_numpy(img).unsqueeze(0).to(args.device)

        # Use the model and visualize the prediction
        with torch.no_grad():
            pred = model(batch)['out']
        pred = torch.softmax(pred, dim=1)
        pred = pred.squeeze(0).cpu().numpy()
        mask = np.argmax(pred, axis=0)
        gt_mask = np.argmax(gt_mask, axis=0)
        # mask = convert_label(mask, inverse=True)
        size = (args.img_size[1], args.img_size[0])
        mask = cv2.resize(mask.astype('float32'), size, interpolation=cv2.INTER_LINEAR).astype('int8')

        # result = convert_color(mask, data_cfg['color_map'])
        result = convert_color(mask, {0: [0, 0, 0], 1: [0, 255, 0], 2: [255, 0, 0]})
        gt_result = convert_color(gt_mask, {0: [0, 0, 0], 1: [0, 255, 0], 2: [255, 0, 0]})
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(img_vis)
        plt.subplot(1, 3, 2)
        plt.imshow(result)
        plt.subplot(1, 3, 3)
        plt.imshow(gt_result)
        plt.show()


if __name__ == '__main__':
    main()
