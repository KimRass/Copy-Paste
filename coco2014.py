import cv2
import numpy as np
from torchvision.datasets import CocoDetection
import random
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import v2
from torchvision import datasets
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
import random
from torch.utils.data import DataLoader


def to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def to_array(image):
    return np.array(image)


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


def vis_label(img, label, palette, beta=0.8):
    # img = to_array(image)
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


def select_objects_randomly(label, select_prob=0.3):
    new_label = list()
    for obj in label:
        if random.random() > select_prob:
            new_label.append(obj)
    return new_label


def get_randomly_selected_obj_mask(img, label, select_prob=0.3):
    h, w = img.shape[:2]
    merged_mask = np.zeros((h, w), dtype=np.uint8)
    new_label = select_objects_randomly(label, select_prob=select_prob)
    for obj in new_label:
        polys = obj["segmentation"]
        mask = get_mask(h=h, w=w, polys=polys)
        merged_mask = np.maximum(merged_mask, mask)
    return new_label, np.stack([merged_mask] * 3, axis=-1)


def merge_two_imgs(img1, img2, mask, transform):
    transformed_img1 = transform(image=img1)["image"]
    transformed = transform(image=img2, mask=mask)
    transformed_img2 = transformed["image"]
    transformed_mask = transformed["mask"]
    vis_label(transformed_img2)
    # to_pil(transformed_mask).show()
    return np.where(transformed_mask == 255, transformed_img2, transformed_img1)


def copy_paste(img1, img2, label1, label2, transform, select_prob):
    select_prob = 0.3
    new_label2, mask = get_randomly_selected_obj_mask(img2, label=label2, select_prob=select_prob)
    merged_img = merge_two_imgs(img1=img1, img2=img2, mask=mask, transform=transform)
    return merged_img


# "We randomly select two images and apply random scale jittering and random horizontal flipping on each of them. Then we select a random subset of objects from one of the images and paste them onto the other image."
# "we remove fully occluded objects and update the masks and bounding boxes of partially occluded objects."
# "For composing new objects into an image, we compute the binary mask (α) of pasted objects using ground-truth annotations and compute the new image as I1 × α + I2 × (1 − α) where I1 is the pasted image and I2 is the main image. To smooth out the edges of the pasted objects we apply a Gaussian filter to α similar to “blending” in [13]. But unlike [13], we also found that simply compos- ing without any blending has similar performance."
if __name__ == "__main__":
    imgs_dir = "/Users/jongbeomkim/Documents/datasets/coco2014/val2014"
    instances_path = "/Users/jongbeomkim/Documents/datasets/coco2014/annotations/instances_val2014.json"
    N_CLASSES = 80
    palette = get_palette(N_CLASSES)

    img_size = 512
    pad_color=(127, 127, 127)
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
        ]
    )
    # ds = CocoDetection(root=imgs_dir, annFile=instances_path, transform=transform)
    ds = CocoDetection(
        root=imgs_dir,
        annFile=instances_path,
        # transforms=lambda image, target: transform(image=to_array(image), target=target),
        transform=lambda x: transform(image=np.array(x))["image"],
        # target_transform=transform,
    )
    dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    di = iter(dl)
    image, target = next(di)
    image.shape
    target
    image1
    label1
    
    image2, label2 = next(di)
    img1 = to_array(image1)
    img2 = to_array(image2)
    vis_label(img2, label2, palette=palette)
    # label1

    SELECT_PROB = 0.6
    img = copy_paste(
        img1=img1,
        img2=img2,
        label1=label1,
        label2=label2,
        transform=transform,
        select_prob=SELECT_PROB,
    )
    to_pil(img).show()


# import torchvision
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# model.eval()