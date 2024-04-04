import cv2
import numpy as np
from torchvision.datasets import CocoDetection
import random
from PIL import Image


def to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def to_array(image):
    return np.array(image)


def get_palette(n_classes):
    rand_perm1 = np.random.permutation(256)[: n_classes]
    rand_perm2 = np.random.permutation(256)[: n_classes]
    rand_perm3 = np.random.permutation(256)[: n_classes]
    return np.stack([rand_perm1, rand_perm2, rand_perm3], axis=1)


def get_color(obj_idx, cls_idx, palette, task):
    if task == "instance":
        return palette[obj_idx]
    elif task == "semantic":
        return palette[cls_idx % len(palette)]


if __name__ == "__main__":
    data_dir = "/home/jbkim/Documents/datasets/val2014"
    instances_file = "/home/jbkim/Documents/datasets/annotations_trainval2014/annotations/instances_val2014.json"

    # 데이터셋 로드
    ds = CocoDetection(root=data_dir, annFile=instances_file)
    di = iter(ds)
    N_CLASSES = 80
    palette = get_palette(N_CLASSES)

    image, label = next(di)
    # image.show()

    img = to_array(image)
    for obj_idx, obj in enumerate(label):
        segmentation = obj["segmentation"]
        bbox = obj["bbox"]
        cls_idx = obj["category_id"]

        bbox = [int(coord) for coord in bbox]

        color = get_color(obj_idx=obj_idx, cls_idx=cls_idx, palette=palette, task="instance")
        if isinstance(segmentation, list):
            h, w = img.shape[:2]

            mask = np.zeros((h, w), dtype=np.uint8)
            for poly in segmentation:
                poly = np.array(poly).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, pts=[poly], color=255) # color is not specified here
            # to_pil(mask).show()

            mask_color = np.stack([mask] * 3, axis=-1)
            mask.shape, mask_color.shape
            for i in range(3):
                mask_color[..., i][mask_color[..., i] == 255] = color[i]

            alpha = ((mask_color > 0).max(axis=2) * 128).astype(np.uint8)
            rgba_mask = np.concatenate([mask_color, alpha[:, :, np.newaxis]], axis=2)

            img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            img_rgba = cv2.addWeighted(img_rgba, 1, rgba_mask, 0.8, 0)

            img = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)

        cls_id = ds.coco.getCatIds(catIds=[cls_idx])
        cls_name = ds.coco.loadCats(cls_id)[0]["name"]
        # cv2.putText(
        #     img, cls_name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        # )
    to_pil(img).show()
