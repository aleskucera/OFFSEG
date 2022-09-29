import os
import cv2
import torch
import albumentations as A
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .utils import visualize, tensor_to_image, mask_to_color, create_dataset, predict_image


def test_model(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model
    model = torch.load('models/Resnet34-0.674.pt')
    model.eval()

    # Load the dataset
    t_test = A.Resize(320, 512, interpolation=cv2.INTER_NEAREST)
    test_set = create_dataset(cfg, 'test', t_test)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=os.cpu_count() // 2)

    for i, (image, mask) in enumerate(test_loader):
        image, mask = image.squeeze(0), mask.squeeze(0)
        image_vis = tensor_to_image(image, cfg.ds.mean, cfg.ds.std)
        mask = mask_to_color(mask, cfg.ds.color_map)

        # Predict the image
        pred_mask = predict_image(model, image, device)
        pred_mask = mask_to_color(pred_mask, cfg.ds.color_map)
        visualize(image=image_vis, ground_truth=mask, prediction=pred_mask)
