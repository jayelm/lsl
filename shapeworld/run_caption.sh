#!/bin/bash

if [ "$#" -eq 1 ]; then
    DATA_DIR="$1"
else
    DATA_DIR="/mnt/fs5/muj/shapeworld_mine_8shot_easy"
fi

if [ "$#" -eq 2 ]; then
    BACKBONE="$2"
else
    BACKBONE="conv4"
fi

python lsl/caption.py --cuda \
    --backbone $BACKBONE \
    --data_dir $DATA_DIR \
    --batch_size 100 \
    --seed $RANDOM \
    exp/cap/"$(basename $DATA_DIR)"_"$BACKBONE"
