import sys
sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/Copy-Paste/")
import json
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
from collections import defaultdict
from torch.utils.data import DataLoader
import torchvision.transforms as T

from coco2014 import to_array, to_pil




annot_path = "/Users/jongbeomkim/Documents/datasets/coco2014/annotations/instances_val2014.json"
img_dir = "/Users/jongbeomkim/Documents/datasets/coco2014/val2014"
with open(annot_path, mode="r") as f:
	annots = json.load(f)

transformer = T.Compose(
    [T.CenterCrop((224)), T.ToTensor(), T.Normalize(mean=0.5, std=0.5)],
)
ds = CocoDetection(img_dir, annot_path, transform=transformer)
dl = DataLoader(ds, batch_size=4, collate_fn=lambda batch: list(zip(*batch)))
image, target = next(iter(dl))
type(target)
len(target)
[{k: v for k, v in t.items()} for t in target[0]]
target[0]


idx = 10
img_id_to_annot_ids = defaultdict(list)
for annot in annots["annotations"]:
    img_id = annot["image_id"]
    annot_id = annot["id"]
    img_id_to_annot_ids[img_id].append(annot_id)
img_ids = list(img_id_to_annot_ids.keys())
img_id = img_ids[idx]
annot_ids = img_id_to_annot_ids[img_id]

img_id_to_img_path = dict()
for img_dict in annots["images"]:
    img_id = img_dict["id"]
    img_name = img_dict["file_name"]
    img_path = Path(img_dir)/img_name
    img_id_to_img_path[img_id] = img_path

img_path = img_id_to_img_path[img_id]
image = Image.open(img_path).convert("RGB")
image.show()


annot_id_to_annot = dict()
for annot in annots["annotations"]:
    annot_id = annot["id"]
    annot_id_to_annot[annot_id] = annot


[annot_id_to_annot[annot_id] for annot_id in annot_ids]








annot = annots["annotations"][idx]
img_id = annot["image_id"]
pref = "COCO_val2014_"
img_path = Path(img_dir)/f"{pref}{str(img_id).zfill(12)}.jpg"

img = to_array(image)

cls_idx = annot["category_id"]

for points in annot["segmentation"]:
    poly = np.array(points).reshape((-1, 2)).astype(np.int32)
    cv2.fillPoly(img, pts=[poly], color=255)
to_pil(img).show()



