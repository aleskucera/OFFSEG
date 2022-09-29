import torch
from torchvision import transforms as T


def predict_image(model, image, device=torch.device('cpu')):
    model.eval()
    model.to(device)
    image = image.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked


def predict_raw_image(model, image, mean, std, device=torch.device('cpu')):
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
