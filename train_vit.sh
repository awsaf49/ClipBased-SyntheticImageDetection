#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
python train.py \
    --batch_size 256 \
    --epochs 20 \
    --lr 2e-4 \
    --min_lr 1e-6 \
    --warmup_epochs 5 \
    --weight_decay 1e-4 \
    --pretrain vitl16_in21k\
    --cropSize 224 \
    --num_workers 4 \
    --wandb_team ece281 \
    --run_name vitl16_in21k \
    --exp_name baseline-v3 \
    --save_dir checkpoints
    # --debug --debug_samples 1000