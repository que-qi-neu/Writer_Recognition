from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def getRGBImage(imageIn):
    image = imageIn
    if isinstance(imageIn, (str, Path)):
            image = Image.open(imageIn).convert("RGB")
    else:
        image = imageIn.convert("RGB")
    return image

def displayImages(image_paths, count=5):
    display_paths = image_paths[:count]
    fig, axes = plt.subplots(1, len(display_paths), figsize=(15, 4))
    if len(display_paths) == 1:
        axes = [axes]

    for ax, path in zip(axes, display_paths):
        img = Image.open(path)
        ax.imshow(img)
        ax.axis('off') 

    plt.tight_layout()
    plt.show()

def modelToGPU(model:nn.modules):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)