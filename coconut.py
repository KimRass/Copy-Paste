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
annot_path = "/home/jbkim/Documents/datasets/coconut/coconut_dataset/annotations/annotations/coconut_s.json"
img_dir = "/home/jbkim/Documents/datasets/coconut/train2017 (1)/train2017"
mask_dir = "/home/jbkim/Documents/datasets/coconut/coconut_dataset/coconut_s/coconut_s/"

with open(annot_path, mode="r") as f:
    coconut = json.load(f)
coconut.keys()
img_dicts = coconut["images"]



annot_dict = defaultdict(str)
annots = coconut["annotations"]
for annot in annots:
    mask_path = Path(mask_dir)/annot["file_name"]
    img_id = annot["image_id"]
    annot_dict[img_id] = str(mask_path)
    pd.DataFrame(annot["segments_info"])


idx = 3
img_dict = img_dicts[idx]
img_id = img_dict["id"]
img_path = Path(img_dir)/img_dict["file_name"]
Image.open(img_path).show()


mask_path = annot_dict[img_id]
mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)[..., 2]
bin_mask = mask[None, ...] == np.unique(mask)[..., None, None]
# Image.fromarray(mask[..., 2] * 20).show()

