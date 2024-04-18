import sys
# sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/Copy-Paste/")
sys.path.insert(0, "/home/jbkim/Desktop/workspace/Copy-Paste")
from coco import LargeScaleJittering, COCODS
from torch.utils.data import DataLoader
from utils import image_to_grid
from copy_paste import CopyPaste


def main():
    # annot_path = "/Users/jongbeomkim/Documents/datasets/coco2014/annotations/instances_val2014.json"
    # img_dir = "/Users/jongbeomkim/Documents/datasets/coco2014/val2014"
    annot_path = "/home/jbkim/Documents/datasets/annotations_trainval2014/annotations/instances_val2014.json"
    img_dir = "/home/jbkim/Documents/datasets/val2014"

    lsj = LargeScaleJittering()
    ds = COCODS(annot_path=annot_path, img_dir=img_dir, transform=lsj)
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=ds.collate_fn)
    di = iter(dl)

    copy_paste = CopyPaste(occ_thresh=0.7, keep_prob=0, select_prob=1)

    image, annots = next(di)
    ds.vis_annots(
        image=image,
        annots=annots,
        labels=False,
        task="instance",
        # alpha=0,
    )

    new_image, new_annots = copy_paste(image, annots)
    ds.vis_annots(
        image=new_image,
        annots=new_annots,
        labels=False,
        task="instance",
        alpha=0,
    )
    ds.vis_annots(
        image=new_image,
        annots=new_annots,
        labels=False,
        task="instance",
    )
    # idx = 3
    # annots["ltrbs"][idx]
    # new_annots["ltrbs"][idx]

if __name__ == "__main__":
    main()
