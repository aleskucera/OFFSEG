#!/usr/bin/env python

import os
import torch
import numpy as np
import torchvision.models.segmentation

from utils import IoU
from tqdm import tqdm
from utils import DiceLoss
from dataset import OFFSEG
from argparse import ArgumentParser
from torch.utils.data import DataLoader


def create_model(architecture, n_inputs, n_outputs, pretrained=True):
    assert architecture in ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101',
                            'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large']

    print('Creating model %s with %i inputs and %i outputs' % (architecture, n_inputs, n_outputs))
    Architecture = eval('torchvision.models.segmentation.%s' % architecture)
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


parser = ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--architecture', type=str, default='fcn_resnet50')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--img_size', nargs='+', default=(320, 512))
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--n_workers', type=int, default=os.cpu_count())
args = parser.parse_args()
args.img_size = tuple(args.img_size)

train_dataset = OFFSEG(crop_size=args.img_size, split='train', size=None)
valid_dataset = OFFSEG(crop_size=args.img_size, split='val', size=None)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers // 2)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.n_workers // 2)

# --------------Load and set model and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using device:', device)

n_classes = len(train_dataset.class_values)

n_inputs = train_dataset[0][0].shape[0]

model = create_model(args.architecture, n_inputs, n_classes, pretrained=False)
model = model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)  # Create adam optimizer

# ----------------Train--------------------------------------------------------------------------
# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
metric_fn = IoU(threshold=0.5, activation='softmax2d')
criterion_fn = DiceLoss(mode='multilabel', from_logits=True, ignore_index=0)

max_metric = -np.Inf
for e in tqdm(range(args.n_epochs)):
    # train epoch
    model = model.train()
    for itr, sample in tqdm(enumerate(train_loader)):
        images, labels = sample
        images, labels = images.to(device), labels.to(device)

        pred = model(images)['out']  # make prediction

        optimizer.zero_grad()
        loss = criterion_fn(pred, labels.long())  # Calculate loss
        loss.backward()  # Backpropagate loss
        optimizer.step()  # Apply gradient descent change to weight

    # validation epoch
    metrics = []
    model = model.eval()
    for itr, sample in tqdm(enumerate(valid_loader)):
        images, labels = sample
        images, labels = images.to(device), labels.int().to(device)
        # convert tensor to int values
        with torch.no_grad():
            pred = model(images)['out']  # make prediction
            metric1 = metric_fn(pred, labels)
        metrics.append(metric1.cpu().numpy())

    metric = np.mean(metrics)

    # save better model
    if max_metric < metric:  # Save model weight
        max_metric = metric
        name = '%s_lr_%g_bs_%d_epoch_%d_%s_iou_%.2f.pth' % \
               (args.architecture,
                args.lr, args.batch_size, e, 'OFFSEG', float(max_metric))
        print("Saving Model:", name)
        torch.save(model, os.path.join(os.path.dirname(__file__), '..', 'models', name))

    print("Epoch: %i" % e)
    print('Train loss: %f' % loss.data.cpu().numpy())
    print('Validation metric: %.3f' % metric)

    if e == 60:
        optimizer.param_groups[0]['lr'] /= 10.0
        print('Decrease decoder learning rate to %f !' % optimizer.param_groups[0]['lr'])
