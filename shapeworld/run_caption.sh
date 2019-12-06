#!/bin/bash

python lsl/caption.py --cuda \
    --backbone vgg16_fixed \
    --batch_size 100 \
    --seed $RANDOM \
    exp/cap/vgg16_fixed
