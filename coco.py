# References:
    # https://www.kaggle.com/code/blondinka/how-to-do-augmentations-for-instance-segmentation

import sys
sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/Copy-Paste/")
import torch
import json
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from torchvision.datasets import CocoDetection
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

from utils import image_to_grid, get_palette, vis_masks
from coco2014 import to_array, to_pil


class LargeScaleJittering(object):
    def __init__(
        self,
        img_size=512,
        pad_color=(127, 127, 127),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=(-0.5, 0.5),
                    scale_limit=(-0.9, 1),
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=pad_color,
                    p=1,
                ),
                A.PadIfNeeded(
                    min_height=img_size,
                    min_width=img_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=pad_color,
                ),
                A.CenterCrop(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    def __call__(self, image, masks):
        return self.transform(image=image, masks=masks)


class COCODS(Dataset):
    def __init__(self, annot_path, img_dir, transform=None):
        self.coco = COCO(annot_path)
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.img_dir = Path(img_dir)

    def __len__(self):
        return len(self.img_ids)

    @staticmethod
    def get_masks(annots, h, w):
        masks = list()
        for annot in annots:
            rles = coco_mask.frPyObjects(annot["segmentation"], h, w)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = mask.any(axis=2).astype(np.uint8)
            masks.append(mask)
        return masks

    @staticmethod
    def get_labels(annots):
        labels = list()
        for annot in annots:
            label = annot["category_id"]
            labels.append(label)
        return labels

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_dicts = self.coco.loadImgs(img_id)
        img_path = str(self.img_dir/img_dicts[0]["file_name"])
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annots = self.coco.loadAnns(ann_ids)
        h, w, _ = img.shape
        masks = self.get_masks(annots=annots, h=h, w=w)

        if self.transform is not None:
            transformed = self.transform(image=img, masks=masks)
            image = transformed["image"]
            masks = transformed["masks"]
            mask = torch.stack(masks, dim=0)

        labels = self.get_labels(annots)
        label = torch.tensor(labels)
        return image, mask, label

    def collate_fn(self, batch):
        images, masks, labels = list(zip(*batch))
        return torch.stack(images, dim=0), masks, labels


if __name__ == "__main__":
    annot_path = "/Users/jongbeomkim/Documents/datasets/coco2014/annotations/instances_val2014.json"
    img_dir = "/Users/jongbeomkim/Documents/datasets/coco2014/val2014"
    # annot_path = "/home/jbkim/Documents/datasets/annotations_trainval2014/annotations/instances_val2014.json"
    # img_dir = "/home/jbkim/Documents/datasets/val2014"

    img_size = 512
    pad_color=(127, 127, 127)
    lsj = LargeScaleJittering()
    ds = COCODS(annot_path=annot_path, img_dir=img_dir, transform=lsj)
    dl = DataLoader(ds, batch_size=4, collate_fn=ds.collate_fn)
    di = iter(dl)
    image, masks, labels = next(di)
    [mask.shape for mask in masks]
    [label.shape for label in labels]

    palette = get_palette(n_classes=80)
    vis_masks(image=image, masks=masks, palette=palette)