import torch
from torch.cuda.amp import GradScaler
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from torchvision.utils import draw_segmentation_masks
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import random
import os
import math


def denorm(x, mean=(0.457, 0.437, 0.404), std=(0.275, 0.271, 0.284)):
    return TF.normalize(
        x, mean=-(np.array(mean) / np.array(std)), std=(1 / np.array(std)),
    )


@torch.inference_mode()
def image_to_grid(image, n_cols, padding=1, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    tensor = image.clone().detach().cpu()
    tensor = denorm(tensor, mean=mean, std=std)
    grid = make_grid(tensor, nrow=n_cols, padding=padding, pad_value=1)
    grid.clamp_(0, 1)
    grid = TF.to_pil_image(grid)
    return grid


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return device


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return device


def get_grad_scaler(device):
    return GradScaler() if device.type == "cuda" else None


def create_dir(x):
    x = Path(x)
    if x.suffix:
        x.parent.mkdir(parents=True, exist_ok=True)
    else:
        x.mkdir(parents=True, exist_ok=True)


def to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def save_image(image, save_path):
    create_dir(save_path)
    to_pil(image).save(str(save_path), quality=100)
    print(f"""Saved image as "{Path(save_path).name}".""")


def get_palette(n_classes):
    rand_perm1 = np.random.permutation(256)[: n_classes]
    rand_perm2 = np.random.permutation(256)[: n_classes]
    rand_perm3 = np.random.permutation(256)[: n_classes]
    return np.stack([rand_perm1, rand_perm2, rand_perm3], axis=1)


def colorize_mask(mask, color):
    colored_mask = np.stack([mask] * 3, axis=-1)
    for i in range(3):
        colored_mask[..., i][colored_mask[..., i] == 255] = color[i]
    return colored_mask


def overlay_mask(img, mask, color, beta=0.8):
    colored_mask = colorize_mask(mask * 255, color=color)
    alpha = ((colored_mask > 0).max(axis=2) * 128).astype(np.uint8)            
    rgba_mask = np.concatenate([colored_mask, alpha[:, :, None]], axis=2)
    rgba_img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    rgba_img = cv2.addWeighted(rgba_img, 1, rgba_mask, beta, gamma=0)
    return cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2RGB)


def overlay_masks(img, mask, palette, beta=0.8):
    new_img = img.copy()
    for mask_idx, gray_mask in enumerate(np.array(mask)):
        color = palette[mask_idx]
        new_img = overlay_mask(img=new_img, mask=gray_mask, color=color, beta=beta)
    return new_img


def to_uint8(image, mean, std):
    return (denorm(image, mean=mean, std=std) * 255).byte()


def vis_masks(
    image,
    masks,
    palette,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    alpha=0.5,
):
    uint8_image = to_uint8(image, mean=mean, std=std)
    images = list()
    for batch_idx in range(image.size(0)):
        drawn_image = draw_segmentation_masks(
            image=uint8_image[batch_idx],
            masks=masks[batch_idx].to(torch.bool),
            alpha=alpha,
            colors=palette[: masks[batch_idx].size(0)].tolist(),
        )
        images.append(drawn_image.cpu())
    
    grid = make_grid(
        torch.stack(images, dim=0),
        nrow=int(image.size(0) ** 0.5),
        padding=1,
        pad_value=255,
    )
    TF.to_pil_image(grid).show()
