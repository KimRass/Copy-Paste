# References:
    # https://www.kaggle.com/code/blondinka/how-to-do-augmentations-for-instance-segmentation

import sys
sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/Copy-Paste/")
# sys.path.insert(0, "/home/jbkim/Desktop/workspace/Copy-Paste")
import torch
from pathlib import Path
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import box_convert
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import COLORS, to_uint8


class LargeScaleJittering(object):
    def __init__(
        self,
        format="coco",
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
            ],
            bbox_params=A.BboxParams(format=format, label_fields=["bbox_ids", "labels"]),
        )

    def __call__(self, image, masks, bboxes, labels):
        return self.transform(
            image=image,
            masks=masks,
            bboxes=bboxes,
            bbox_ids=range(len(bboxes)),
            labels=labels,
        )


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
    def get_coco_bboxes(annots):
        return [annot["bbox"] for annot in annots]

    @staticmethod
    def get_labels(annots):
        return [annot["category_id"] for annot in annots]

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
        coco_bboxes = self.get_coco_bboxes(annots)
        labels = self.get_labels(annots)

        if self.transform is not None:
            transformed = self.transform(
                image=img, masks=masks, bboxes=coco_bboxes, labels=labels,
            )
            image = transformed["image"]
            masks = transformed["masks"]
            coco_bboxes = transformed["bboxes"]
            bbox_ids = transformed["bbox_ids"]
            labels = transformed["labels"]

        return (
            image,
            torch.stack([masks[bbox_id] for bbox_id in bbox_ids], dim=0),
            torch.tensor(coco_bboxes),
            torch.tensor(labels),
        )

    def collate_fn(self, batch):
        images, masks, coco_bboxes, labels = list(zip(*batch))
        ltrbs = [
            box_convert(boxes=coco_bbox, in_fmt="xywh", out_fmt="xyxy")
            for coco_bbox
            in coco_bboxes
        ]
        annots = {"masks": masks, "ltrbs": ltrbs, "labels": labels}
        return torch.stack(images, dim=0), annots

    def labels_to_class_names(self, labels):
        return [[self.coco.cats[j]["name"] for j in i.tolist()] for i in labels]

    def vis_annots(
        self,
        image,
        annots,
        task="instance",
        colors=COLORS,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        alpha=0.6,
    ):
        uint8_image = to_uint8(image.cpu(), mean=mean, std=std)
        class_names = self.labels_to_class_names(annots["labels"])
        images = list()
        for batch_idx in range(image.size(0)):
            if task == "instance":
                picked_colors = [
                    colors[i % len(colors)] for i in annots["labels"][batch_idx].tolist()
                ]
            elif task == "semantic":
                picked_colors = colors

            new_image = uint8_image[batch_idx]
            new_image = draw_segmentation_masks(
                image=new_image,
                masks=annots["masks"][batch_idx].to(torch.bool),
                alpha=alpha,
                colors=picked_colors,
            )
            new_image = draw_bounding_boxes(
                image=new_image,
                boxes=annots["ltrbs"][batch_idx],
                labels=class_names[batch_idx],
                colors=picked_colors,
                width=2,
                # font=Path(__file__).resolve().parent/"resources/NotoSans_Condensed-Medium.ttf",
                font="/Users/jongbeomkim/Desktop/workspace/Copy-Paste/resources/NotoSans_Condensed-Medium.ttf",
                font_size=14,
            )
            images.append(new_image)

        grid = make_grid(
            torch.stack(images, dim=0),
            nrow=int(image.size(0) ** 0.5),
            padding=1,
            pad_value=255,
        )
        TF.to_pil_image(grid).show()


if __name__ == "__main__":
    annot_path = "/Users/jongbeomkim/Documents/datasets/coco2014/annotations/instances_val2014.json"
    img_dir = "/Users/jongbeomkim/Documents/datasets/coco2014/val2014"
    # annot_path = "/home/jbkim/Documents/datasets/annotations_trainval2014/annotations/instances_val2014.json"
    # img_dir = "/home/jbkim/Documents/datasets/val2014"

    img_size = 512
    pad_color=(127, 127, 127)
    lsj = LargeScaleJittering()
    ds = COCODS(annot_path=annot_path, img_dir=img_dir, transform=lsj)
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=ds.collate_fn)
    di = iter(dl)

    image, annots = next(di)
    # [mask.size(0) for mask in masks]
    # [ltrb.size(0) for ltrb in ltrbs]
    # [label.size(0) for label in labels]

    ds.vis_annots(
        image=image,
        annots=annots,
        task="semantic",
        # task="instance",
    )
