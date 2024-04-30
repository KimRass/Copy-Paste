from torch.utils.data import DataLoader
from pathlib import Path
import argparse

from coco import COCODS
from lsj import LargeScaleJittering
from copy_paste import CopyPaste


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--annot_path", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False, default=16)
    parser.add_argument("--num_samples", type=int, required=False, default=10)

    args = parser.parse_args()

    args_dict = vars(args)
    new_args_dict = dict()
    for k, v in args_dict.items():
        new_args_dict[k.upper()] = v
    args = argparse.Namespace(**new_args_dict)
    return args


def main():
    args = get_args()
    SAMPLES_DIR = Path(__file__).resolve().parent/"samples"

    lsj = LargeScaleJittering()
    copy_paste = CopyPaste(occ_thresh=0.7, keep_prob=0.5, select_prob=0.5)
    ds = COCODS(annot_path=args.ANNOT_PATH, img_dir=args.IMG_DIR, transform=lsj)
    dl = DataLoader(
        ds, batch_size=args.BATCH_SIZE, shuffle=False, collate_fn=ds.collate_fn,
    )
    for batch_idx, (image, annots) in enumerate(dl):
        vis_bef = ds.vis_annots(
            image=image,
            annots=annots,
            task="instance",
        )
        vis_bef.save(SAMPLES_DIR/f"{batch_idx}-original.jpg")
        new_image, new_annots = copy_paste(image, annots)
        vis_aft = ds.vis_annots(
            image=new_image,
            annots=new_annots,
            task="instance",
        )
        vis_aft.save(SAMPLES_DIR/f"{batch_idx}-copy_paste.jpg")
        if batch_idx >= args.NUM_SAMPLES:
            break


if __name__ == "__main__":
    main()
