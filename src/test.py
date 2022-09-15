import cv2
import torch
import numpy as np
import albumentations as A
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset import OFFSEGDataset
from torchvision import transforms as T
from parameter_parser import ParametersImage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model():
    # Load the model
    model = torch.load('Unet-Mobilenet_v2_mIoU-0.688.pt')
    model.eval()

    # Load the dataset
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t_test = A.Resize(320, 512, interpolation=cv2.INTER_NEAREST)
    dataset = OFFSEGDataset(split='test', mean=mean, std=std, transform=t_test)

    # Load the image
    image, label = dataset[0]

    # Predict the image
    pred_mask, score = predict_image_mask_miou(model, image, label)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(image)
    ax1.set_title('Picture')

    ax2.imshow(label)
    ax2.set_title('Ground truth')
    ax2.set_axis_off()

    ax3.imshow(pred_mask)
    ax3.set_title('UNet-MobileNet | mIoU {:.3f}'.format(score))
    ax3.set_axis_off()
    plt.show()


def predict_image_mask_miou(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score


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


if __name__ == '__main__':
    test_model()
