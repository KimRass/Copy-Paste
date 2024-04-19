#!/bin/sh

source set_pythonpath.sh

python3 ../vis.py\
    --annot_path="/Users/jongbeomkim/Documents/datasets/coco2014/annotations/instances_val2014.json"\
    --img_dir="/Users/jongbeomkim/Documents/datasets/coco2014/val2014"\
    --batch_size=16\
    --num_samples=10
