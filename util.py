import os
import torch
from torchvision.utils import save_image
import math


def accu(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target = target
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _save_image(img_dir, image, name):
    file_name = img_dir + '/' + name + '.jpg'
    nrow = int(math.sqrt(image.shape[0]))
    save_image(image.cpu(), file_name, nrow=nrow, padding=2, normalize=True)
    return
