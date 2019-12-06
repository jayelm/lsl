#!/bin/bash

BACKBONE="conv4"

python lsl/caption.py --cuda \
    --backbone $BACKBONE \
    --batch_size 100 \
    --seed $RANDOM \
    exp/cap/$BACKBONE
