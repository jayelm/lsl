#!/bin/bash

python lsl/train.py --cuda \
    --infer_hyp \
    --hypo_lambda 1.0 \
    --batch_size 100 \
    --seed $RANDOM \
    exp/l3
