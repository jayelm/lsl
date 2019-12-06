#!/bin/bash

python lsl/caption.py --cuda \
    --backbone conv4 \
    --batch_size 100 \
    --seed $RANDOM \
    exp/cap/meta
