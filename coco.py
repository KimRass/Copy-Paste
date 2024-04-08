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

from coco2014 import to_array, to_pil, get_palette

get_palette(n_classes=3)


# class COCODS(Dataset):
#     def __init__(self, annot_path, img_dir, transform):
#         with open(annot_path, mode="r") as f:
#             self.annots = json.load(f)

#         img_id_to_img_path = dict()
#         for img_dict in self.annots["images"]:
#             img_id = img_dict["id"]
#             img_name = img_dict["file_name"]
#             img_path = Path(img_dir)/img_name
#             img_id_to_img_path[img_id] = str(img_path)

#         self.img_path_to_annots = defaultdict(list)
#         for annot in self.annots["annotations"]:
#             img_id = annot["image_id"]
#             img_path = img_id_to_img_path[img_id]
#             self.img_path_to_annots[img_path].append(annot)
#         self.img_paths = list(self.img_path_to_annots.keys())

#         self.transform = transform

#     def __len__(self):
#         return len(self.img_path_to_annots)

#     @staticmethod
#     def annot_to_mask(h, w, annot):
#         mask = np.zeros((h, w), dtype=np.uint8)
#         for points in annot["segmentation"]:
#             # print(np.array(points).shape)
#             poly = np.array(points).reshape((-1, 2)).astype(np.int32)
#             cv2.fillPoly(mask, pts=[poly], color=255)
#         return mask

#     def annots_to_mask(self, h, w, annots):
#         masks = list()
#         for annot in annots:
#             print(annot.keys())
#             mask = self.annot_to_mask(h=h, w=w, annot=annot)
#             masks.append(mask)
#         return torch.from_numpy(np.stack(masks, axis=0))

#     def __getitem__(self, idx):
#         # idx = 10
#         img_path = self.img_paths[idx]
#         image = Image.open(img_path).convert("RGB")
#         image.show()
#         h, w = image.size
#         annots = self.img_path_to_annots[img_path]
#         # print(img_path)
#         # print(annots)
#         mask = self.annots_to_mask(h=h, w=w, annots=annots)
#         image = self.transform(image)
#         # print(image.shape, )
#         return image, {"mask": mask}

#     def collate_fn(self, batch):
#         images = list()
#         for image, annot in batch:
#             images.append(image)
#         return torch.stack(images, dim=0), annot


class COCODS(Dataset):
    def __init__(self, annot_path, img_dir, transform=None):
        self.coco = COCO(annot_path)
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.img_dir = Path(img_dir)

    def __len__(self):
        return len(self.img_ids)

    @staticmethod
    def convert_coco_poly_to_mask(annots, h, w):
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
    def annot_to_mask(h, w, annot):
        mask = np.zeros((h, w), dtype=np.uint8)
        for points in annot["segmentation"]:
            # print(np.array(points).shape)
            poly = np.array(points).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, pts=[poly], color=255)
        return mask

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_dicts = self.coco.loadImgs(img_id)
        img_path = str(self.img_dir/img_dicts[0]["file_name"])
        # image = Image.open(img_path).convert("RGB")
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annots = self.coco.loadAnns(ann_ids)
        h, w, _ = img.shape
        masks = self.convert_coco_poly_to_mask(annots=annots, h=h, w=w)

        if self.transform is not None:
            transformed = self.transform(image=img, masks=masks)
            img = transformed["image"]
            masks = transformed["masks"]
        return img, masks


    def collate_fn(self, batch):
        images = list()
        mask_tensors = list()
        for image, masks in batch:
            images.append(image)
            mask_tensors.append(torch.stack(masks, dim=0))
        return torch.stack(images, dim=0), mask_tensors


if __name__ == "__main__":
    # annot_path = "/Users/jongbeomkim/Documents/datasets/coco2014/annotations/instances_val2014.json"
    annot_path = "/home/jbkim/Documents/datasets/annotations_trainval2014/annotations/instances_val2014.json"
    # img_dir = "/Users/jongbeomkim/Documents/datasets/coco2014/val2014"
    img_dir = "/home/jbkim/Documents/datasets/val2014"

    img_size = 512
    pad_color=(127, 127, 127)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = A.Compose(
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
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    ds = COCODS(annot_path=annot_path, img_dir=img_dir, transform=transform)
    image, masks = ds[5]
    image.shape, [mask.shape for mask in masks]
