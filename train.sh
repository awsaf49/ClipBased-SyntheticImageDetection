#!/bin/bash

python train.py \
    --batch_size 128 \
    --epochs 20 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --pretrain clipL14commonpool \
    --num_workers 4 \
    --save_dir checkpoints