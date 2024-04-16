import sys
# sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/Copy-Paste/")
sys.path.insert(0, "/home/jbkim/Desktop/workspace/Copy-Paste")
import torch


def get_select_mask(mask2, prob=0.5):
    return torch.rand(mask2.size(0)) < prob


def get_copy_paste_image(image1, image2, mask2):
    cat_mask = torch.any(mask2, dim=0)[None, ...].repeat(3, 1, 1)
    return cat_mask * image2 + (1 - cat_mask) * image1


def perform_copy_paste(image, annots, idx1, idx2):
    masks = annots["masks"]
    mask1 = masks[idx1]
    mask2 = masks[idx2]
    select_mask = get_select_mask(mask2)
    mask2 = mask2[select_mask]

    labels = annots["labels"]
    label1 = labels[idx1]
    label2 = labels[idx2][select_mask]

    ltrbs = annots["ltrbs"]
    ltrb1 = ltrbs[idx1]
    ltrb2 = ltrbs[idx2][select_mask]

    image1 = image[idx1]
    image2 = image[idx2]

    image[idx1] = get_copy_paste_image(image1=image1, image2=image2, mask2=mask2)
    annots["masks"] = list(annots["masks"])
    annots["masks"][idx1] = torch.cat([mask1, mask2], dim=0)
    annots["ltrbs"][idx1] = torch.cat([ltrb1, ltrb2], dim=0)
    annots["labels"] = list(annots["labels"])
    annots["labels"][idx1] = torch.cat([label1, label2], dim=0)
    return image, annots


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
        # task="semantic",
        task="instance",
    )


    idx1 = 1
    idx2 = 2
    prob = 0.5
    new_image, new_annots = perform_copy_paste(image, annots, idx1=1, idx2=2)
    annots["masks"][idx1].shape, new_annots["masks"][idx1].shape
    ds.vis_annots(
        image=new_image,
        annots=new_annots,
        task="instance",
        alpha=0.3,
    )
