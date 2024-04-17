import sys
# sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/Copy-Paste/")
sys.path.insert(0, "/home/jbkim/Desktop/workspace/Copy-Paste")
import torch
import random


class CopyPaste(object):
    """
    "We randomly select two images and apply random scale jittering and random
    horizontal flipping on each of them. Then we select a random subset of objects
    from one of the images and paste them onto the other image."
    """
    def __init__(self, keep_prob=0.7, select_prob=0.5, occ_thresh=0.3):
        self.keep_prob = keep_prob
        self.select_prob = select_prob
        self.occ_thresh = occ_thresh

    def get_select_mask(self, mask2):
        return torch.rand(mask2.size(0)) < self.select_prob

    def exclude_occluded_objects(self, mask1, mask2):
        """
        "We remove fully occluded objects and update the masks and bounding boxes
        of partially occluded objects."
        """
        area = torch.sum(mask1, dim=(1, 2))
        intersec = torch.sum(
            mask1 * torch.any(mask2, dim=0, keepdim=True), dim=(1, 2),
        )
        ioa = intersec / area
        new_mask1 = mask1[ioa > self.occ_thresh]
        if new_mask1.nelement() == 0:
            return torch.cat([new_mask1, mask2], dim=0)
        else:
            return mask2

    @staticmethod
    def get_copy_paste_image(image1, image2, mask2):
        """
        "We compute the binary mask ($\alpha$) of pasted objects using ground-truth
        annotations and compute the new image as
        $I_{1} \times \alpha + I_{2} \times (1 - \alpha)$ where $I_{1}$ is
        the pasted image and $I_{2}$ is the main image."
        "To smooth out the edges of the pasted objects we apply a Gaussian filter
        to \alpha similar to “blending” in [13]. Simply composing without any blending has similar performance."
        """
        cat_mask = torch.any(mask2, dim=0, keepdim=True).repeat(3, 1, 1)
        return cat_mask * image2 + (1 - cat_mask) * image1

    def apply(self, image, annots, idx1, idx2):
        image1 = image[idx1]

        masks = annots["masks"]
        mask1 = masks[idx1]

        labels = annots["labels"]
        label1 = labels[idx1]

        ltrbs = annots["ltrbs"]
        ltrb1 = ltrbs[idx1]
        if idx1 == idx2:
            return image1, mask1, label1, ltrb1
        else:
            image2 = image[idx2]
            mask2 = masks[idx2]
            select_mask = self.get_select_mask(mask2)
            mask2 = mask2[select_mask]
            label2 = labels[idx2][select_mask]
            ltrb2 = ltrbs[idx2][select_mask]
            return (
                self.get_copy_paste_image(
                    image1=image1, image2=image2, mask2=mask2,
                ),
                self.exclude_occluded_objects(mask1=mask1, mask2=mask2),
                torch.cat([label1, label2], dim=0),
                torch.cat([ltrb1, ltrb2], dim=0),
            )

    def __call__(self, image, annots):
        batch_size = image.size(0)
        ls1 = list(range(batch_size))
        ls2 = list(range(batch_size))
        random.shuffle(ls2)
        print(ls2)

        image_tensors = list()
        masks = list()
        labels = list()
        ltrbs = list()
        for idx1, idx2 in zip(ls1, ls2):
            if random.random() < self.keep_prob:
                idx2 = idx1

            image_tensor, mask, label, ltrb = self.apply(
                image, annots, idx1=idx1, idx2=idx2,
            )
            image_tensors.append(image_tensor)
            masks.append(mask)
            labels.append(label)
            ltrbs.append(ltrb)
        return (
            torch.stack(image_tensors, dim=0),
            {
                "masks": masks,
                "labels": labels,
                "ltrbs": ltrbs,
            },
        )


if __name__ == "__main__":
    from coco import LargeScaleJittering, COCODS
    from torch.utils.data import DataLoader

    # annot_path = "/Users/jongbeomkim/Documents/datasets/coco2014/annotations/instances_val2014.json"
    # img_dir = "/Users/jongbeomkim/Documents/datasets/coco2014/val2014"
    annot_path = "/home/jbkim/Documents/datasets/annotations_trainval2014/annotations/instances_val2014.json"
    img_dir = "/home/jbkim/Documents/datasets/val2014"

    img_size = 512
    pad_color=(127, 127, 127)
    lsj = LargeScaleJittering()
    ds = COCODS(annot_path=annot_path, img_dir=img_dir, transform=lsj)
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=ds.collate_fn)
    di = iter(dl)

    image, annots = next(di)

    ds.vis_annots(
        image=image,
        annots=annots,
        labels=False,
        task="instance",
        alpha=0.6,
    )


    copy_paste = CopyPaste(occ_thresh=0.3, keep_prob=0, select_prob=1)

    new_image, new_annots = copy_paste(image, annots)
    new_annots["masks"][1].nelement() == 0
    ds.vis_annots(
        image=new_image,
        annots=new_annots,
        labels=False,
        task="instance",
        alpha=0.6,
    )

    # mask1 = annots["masks"][3][:4, ...]
    # mask2 = annots["masks"][3][4:, ...]
    # mask1.shape, mask2.shape
    # area = torch.sum(mask1, dim=(1, 2))
    # intersec = torch.sum(mask1 * torch.any(mask2, dim=0, keepdim=True), dim=(1, 2))
    # ioa = intersec / area
    # occ_thresh = 0.3
    # mask1 = mask1[ioa > occ_thresh]
    
    # mask2.max()