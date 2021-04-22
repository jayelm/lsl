#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python -u lsl/train.py --cuda \
    --batch_size 64 \
    --seed $RANDOM \
    --lr 1e-6 \
    --epochs 150 \
    --backbone lxmert \
    --optimizer bertadam \
    exp/meta > debug.out
