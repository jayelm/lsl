#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -u lsl/train.py --cuda \
    --infer_hyp \
    --retrive_hint \
    --hypo_lambda 1.0 \
    --batch_size 100 \
    --seed $RANDOM \
    exp/l3/dot_prod
