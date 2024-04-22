# Sources:
    # https://www.kaggle.com/datasets/xueqingdeng/coconut/
    # https://github.com/bytedance/coconut_cvpr2024?tab=readme-ov-file

from coco import COCODS


if __name__ == "__main__":
    from pycocotools.coco import COCO
    import json
    from pathlib import Path
    from PIL import Image
    import numpy as np
    import cv2
    import einops
    annot_path = "/home/jbkim/Documents/datasets/coconut/coconut_dataset/annotations/annotations/coconut_s.json"
    img_dir = "/home/jbkim/Documents/datasets/coconut/train2017 (1)/train2017"
    mask_dir = "/home/jbkim/Documents/datasets/coconut/coconut_dataset/coconut_s/coconut_s/"
    
    with open(annot_path, mode="r") as f:
        coconut = json.load(f)
    coconut.keys()
    img_dicts = coconut["images"]
    
    
    
    
    idx = 3
    img_dict = img_dicts[idx]
    img_id = img_dict["id"]
    img_path = Path(img_dir)/img_dict["file_name"]

    annots = coconut["annotations"]
    for annot in annots:
        mask_path = Path(mask_dir)/annot["file_name"]
        img_id = annot["image_id"]
        # break
    annots[0]
    
    
    mask_path = "/home/jbkim/Documents/datasets/coconut/coconut_dataset/coconut_s/coconut_s/000000000009.png"
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    mask.shape
    np.unique(mask)
    np.unique(mask[..., 0])
    mask = mask[..., 0][None, ...]
    # mask.shape
    _, h, w = mask.shape
    
    repeated_mask = einops.repeat(
        np.unique(mask), pattern="i -> i h w", h=h, w=w,
    )
    repeated_mask.shape
    (repeated_mask == mask).shape
