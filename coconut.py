# Sources:
    # https://www.kaggle.com/datasets/xueqingdeng/coconut/
    # https://github.com/bytedance/coconut_cvpr2024

from coco import COCODS
from pycocotools.coco import COCO
import json
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import einops
import torch
from collections import defaultdict
import pandas as pd
import torch
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from torch.utils.data import Dataset
from torchvision.utils import (
    make_grid,
    draw_segmentation_masks,
    draw_bounding_boxes,
)
from torchvision.ops import box_convert
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import numpy as np

from lsj import LargeScaleJittering


class CocoNutDS(Dataset):
    def __init__(
        self, annot_path, img_dir, mask_dir, img_size=512, transform=None,
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        with open(annot_path, mode="r") as f:
            self.coconut = json.load(f)

        self.img_id_to_annot_info = dict()
        for annot_info in self.coconut["annotations"]:
            img_id = annot_info["image_id"]
            self.img_id_to_annot_info[img_id] = annot_info

    def __len__(self):
        return len(self.coconut["images"])

    def __getitem__(self, idx):
        img_info = self.coconut["images"][idx]
        img_id = img_info["id"]
        img_path = Path(self.img_dir)/img_info["file_name"]
        img = cv2.imread(str(img_path), flags=cv2.IMREAD_COLOR)
        annot_info = self.img_id_to_annot_info[img_id]
        label = torch.tensor(
            [
                info["category_id"]
                for idx, info
                in enumerate(annot_info["segments_info"], start=1)
                if idx == info["id"]
            ],
            dtype=torch.long
        )
        mask_path = Path(self.mask_dir)/annot_info["file_name"]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)[..., 2]
        mask = mask[None, ...] == np.unique(mask)[..., None, None]

        if self.transform is not None:
            transformed = self.transform(
                image=img, masks=masks, bboxes=coco_bboxes, labels=labels,
            )
        return img, mask[1:, ...], label


if __name__ == "__main__":
    annot_path = "/home/jbkim/Documents/datasets/coconut/coconut_dataset/annotations/annotations/coconut_s.json"
    img_dir = "/home/jbkim/Documents/datasets/coconut/train2017 (1)/train2017"
    mask_dir = "/home/jbkim/Documents/datasets/coconut/coconut_dataset/coconut_s/coconut_s/"
    coconut_ds = CocoNutDS(
        annot_path=annot_path, img_dir=img_dir, mask_dir=mask_dir,
    )
    for idx in range(1000):
        img, mask, label = coconut_ds[idx]
        mask.shape, label.shape
