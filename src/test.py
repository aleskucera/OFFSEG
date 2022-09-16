import os
import cv2
import torch
import numpy as np
import albumentations as A

from dataset import OFFSEGDataset
from torchvision import transforms as T
from utils import mIoU, convert_color, visualize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from config import Config

cfg = Config()


def test_model():
    # Load the model
    model = torch.load('../models/Unet-Mobilenet_v2_mIoU-0.773.pt')
    model.eval()

    # Load the dataset
    t_test = A.Resize(320, 512, interpolation=cv2.INTER_NEAREST)
    dataset = OFFSEGDataset(split='test', transform=t_test)

    for i in range(5):
        idx = np.random.randint(0, len(dataset))
        image, label = dataset[idx]
        label = convert_color(label, cfg['dataset_color_map'])

        # Predict the image
        pred_mask = predict_image(model, image, mean=cfg['dataset_mean'], std=cfg['dataset_std'])
        pred_mask = convert_color(pred_mask, cfg['dataset_color_map'])
        visualize(image=image, ground_truth=label, prediction=pred_mask)


def predict_image(model, image, mean=(0.44833934, 0.49257269, 0.46350682),
                  std=(0.22696872, 0.23755784, 0.27277329)):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked


if __name__ == '__main__':
    test_model()
