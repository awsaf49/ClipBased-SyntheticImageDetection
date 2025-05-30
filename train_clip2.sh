#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python train.py \
    --batch_size 256 \
    --epochs 15 \
    --lr 2e-4 \
    --min_lr 1e-6 \
    --warmup_epochs 5 \
    --weight_decay 1e-4 \
    --pretrain clipL14commonpool \
    --cropSize 224 \
    --num_workers 4 \
    --wandb_team ece281 \
    --run_name clipL14commonpool \
    --exp_name baseline-v3 \
    --save_dir checkpoints
    # --debug --debug_samples 1000