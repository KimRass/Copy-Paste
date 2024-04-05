import cv2
import numpy as np
from torchvision.datasets import CocoDetection
import random
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import v2
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
import random


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
        # return palette[cls_idx % len(palette)]
        return palette[cls_idx]


def get_mask(h, w, polys):
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        poly = np.array(poly).reshape((-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, pts=[poly], color=255)
    return mask


def colorize_mask(mask, color):
    colored_mask = np.stack([mask] * 3, axis=-1)
    for i in range(3):
        colored_mask[..., i][colored_mask[..., i] == 255] = color[i]
    return colored_mask


def overlay_mask(img, colored_mask, beta):
    alpha = ((colored_mask > 0).max(axis=2) * 128).astype(np.uint8)
    rgba_mask = np.concatenate([colored_mask, alpha[:, :, None]], axis=2)
    rgba_img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    rgba_img = cv2.addWeighted(rgba_img, 1, rgba_mask, beta, gamma=0)
    return cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2RGB)


def put_cls_name(img, cls_name, poly, color):
    # cls_id = ds.coco.getCatIds(catIds=[cls_idx])
    # cls_name = ds.coco.loadCats(cls_id)[0]["name"]
    x, y = np.mean(poly, axis=0).astype(np.int32)
    cv2.putText(
        img,
        text=cls_name,
        org=(x, y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=color.tolist(),
        thickness=1,
    )


def vis_label(image, label, palette, beta=0.8):
    img = to_array(image)
    for obj_idx, obj in enumerate(label):
        polys = obj["segmentation"]
        cls_idx = obj["category_id"]

        if isinstance(polys, list):
            h, w = img.shape[:2]
            mask = get_mask(h=h, w=w, polys=polys)
            color = get_color(
                obj_idx=obj_idx, cls_idx=cls_idx, palette=palette, task="instance",
            )
            colored_mask = colorize_mask(mask, color=color)
            img = overlay_mask(img, colored_mask=colored_mask, beta=beta)
    to_pil(img).show()


class LargeScaleJittering(DualTransform):
    def __init__(self, img_size, pad_color=(127, 127, 127), always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

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
            ]
        )

    # def apply(self, img, **params):
    #     augmented = self.transform(image=img, **params)
    #     return augmented['image']
    def apply(self, img, mask=None, **params):
        if mask is not None:
            return self.transforms(image=img, mask=mask, **params)
        else:
            return self.transforms(image=img, **params)

    # def apply_to_mask(self, mask, **params):
    #     augmented = self.transform(image=mask, **params)
    #     return augmented['image']

    def get_params_dependent_on_targets(self, params):
        return {}


# "We randomly select two images and apply random scale jittering and random horizontal flipping on each of them. Then we select a random subset of objects from one of the images and paste them onto the other image."
# "we remove fully occluded objects and update the masks and bounding boxes of partially occluded objects."
# "For composing new objects into an image, we compute the binary mask (α) of pasted objects using ground-truth annotations and compute the new image as I1 × α + I2 × (1 − α) where I1 is the pasted image and I2 is the main image. To smooth out the edges of the pasted objects we apply a Gaussian filter to α similar to “blending” in [13]. But unlike [13], we also found that simply compos- ing without any blending has similar performance."
if __name__ == "__main__":
    data_dir = "/home/jbkim/Documents/datasets/val2014"
    instances_file = "/home/jbkim/Documents/datasets/annotations_trainval2014/annotations/instances_val2014.json"

    # 데이터셋 로드
    ds = CocoDetection(root=data_dir, annFile=instances_file)
    di = iter(ds)
    N_CLASSES = 80
    palette = get_palette(N_CLASSES)



    image, label = next(di)
    # new_image = vis_label(image, label=label, palette=palette)

    img_size = 512
    transform = LargeScaleJittering(img_size=img_size)

    image1, label1 = next(di)

    # image1.show()
    img1 = to_array(image1)
    img1 = transform(image=img1)["image"]
    img1.shape

    image2, label2 = next(di)
    
    img2 = to_array(image2)
    prob = 0.3
    h, w = img2.shape[:2]
    new_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for obj_idx, obj in enumerate(label2):
        if random.random() > prob:
            continue
        polys = obj["segmentation"]
        cls_idx = obj["category_id"]
        mask = get_mask(h=h, w=w, polys=polys)
        mask = np.stack([mask] * 3, axis=-1)

        new_mask = np.maximum(new_mask, mask)

    transformed2 = transform(image=img2, mask=new_mask)
    img2 = transformed2["image"]
    new_mask = transformed2["mask"]
    # to_pil(img2).show()

    new_img = np.where(new_mask == 255, img2, img1)
    to_pil(new_img).show()
    to_pil(new_mask).show()
    # to_pil(img1).show()





    class CustomTransform(A.DualTransform):
        def __init__(self, img_size, pad_color=(0, 0, 0), always_apply=False, p=1.0):
            super().__init__(always_apply=always_apply, p=p)
            
            self.transforms = A.Compose([
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
            ])

        def apply(self, img, mask=None, **params):
            if mask is not None:
                return self.transforms(image=img, mask=mask, **params)
            else:
                return self.transforms(image=img, **params)

        def get_transform_init_args_names(self):
            return ('img_size', 'pad_color')
        
    img_size = 512
    tt = CustomTransform(img_size = 512)
    img1 = to_array(image1)
    new_mask = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    out = tt(image=img1, mask=new_mask)
    out = tt(image=img1)
    # out.keys()