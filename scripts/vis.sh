#!/bin/sh

source set_pythonpath.sh

python3 ../vis.py\
    --annot_path="/home/jbkim/Documents/datasets/annotations_trainval2014/annotations/instances_val2014.json"\
    --img_dir="/home/jbkim/Documents/datasets/val2014"
