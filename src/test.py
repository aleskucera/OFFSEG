import os
import cv2
import torch
import albumentations as A
from torchvision import transforms as T
from torch.utils.data import DataLoader, ConcatDataset

from utils import visualize, tensor_to_image, mask_to_color
from dataset import RellisDataset, RugdDataset, CityscapesDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from config import Config

cfg = Config()


def create_dataset(rellis_path, rugd_path, cityscapes_path, split, size=None, transform=None):
    rellis_dataset = RellisDataset(path=rellis_path, split=split, size=size, transform=transform)
    rugd_dataset = RugdDataset(path=rugd_path, split=split, size=size, transform=transform)
    cityscapes_dataset = CityscapesDataset(path=cityscapes_path, split=split, size=size, transform=transform)
    return ConcatDataset([rellis_dataset, rugd_dataset, cityscapes_dataset])


def test_model(p: dict):
    # Load the model
    model = torch.load('../models/Resnet34.pt')
    model.eval()

    # Load the dataset
    t_test = A.Resize(320, 512, interpolation=cv2.INTER_NEAREST)
    test_set = create_dataset(p['rellis_path'], p['rugd_path'], p['cityscapes_path'],
                              'test', transform=t_test)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=os.cpu_count() // 2)

    for i, (image, mask) in enumerate(test_loader):
        image, mask = image.squeeze(0), mask.squeeze(0)
        image_vis = tensor_to_image(image, cfg['dataset_mean'], cfg['dataset_std'])
        mask = mask_to_color(mask, cfg['priseg_dataset'])

        # Predict the image
        pred_mask = predict_image(model, image)
        pred_mask = mask_to_color(pred_mask, cfg['priseg_dataset'])
        visualize(image=image_vis, ground_truth=mask, prediction=pred_mask)


def test2(p: dict):
    # Load the model
    model = torch.load('../models/Resnet34.pt')
    model.eval()


def predict_image(model, image):
    model.eval()
    model.to(device)
    image = image.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked


def predict_raw_image(model, image, mean, std):
    model.eval()
    model.to(device)
    # Normalize image
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    image = image.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked
