#!/bin/bash

BACKBONE="conv4"
DATA_DIR="/mnt/fs5/muj/shapeworld_mine_8shot_easy"

python lsl/caption.py --cuda \
    --backbone $BACKBONE \
    --data_dir $DATA_DIR \
    --batch_size 100 \
    --seed $RANDOM \
    exp/cap/"$(basename $DATA_DIR)"_"$BACKBONE"
