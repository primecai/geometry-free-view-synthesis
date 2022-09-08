#!/bin/bash -l

echo Launching experiment $1
echo GPU $2

# CUDA_VISIBLE_DEVICES=$2 nohup python scripts/sparse_from_realestate_format.py --txt_src /media/data6/shengqu/datasets/acid_single/train/cameras/ --img_src /media/data6/shengqu/datasets/acid_single/train/frames/ --spa_dst /media/data6/shengqu/datasets/acid_single_sparse/train \
# > /media/data6/shengqu/datasets/acid_single_sparse/log.out 2>&1 &
CUDA_VISIBLE_DEVICES=$2 nohup python warp_single_image.py \
> /media/data6/shengqu/datasets/acid_single_seq/log.out 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python warp_single_image.py
echo DETACH
