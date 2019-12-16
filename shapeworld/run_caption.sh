#!/bin/bash

if [ "$#" -ge 1 ]; then
    DATA_DIR="$1"
else
    DATA_DIR="/mnt/fs5/muj/shapeworld_mine_8shot_easy"
fi

if [ "$#" -ge 2 ]; then
    BACKBONE="$2"
else
    BACKBONE="conv4"
fi

if [ "$BACKBONE" = "resnet18" ]; then
    BATCH_SIZE="32"
else
    BATCH_SIZE="100"
fi

python lsl/caption.py --cuda \
    --backbone $BACKBONE \
    --data_dir $DATA_DIR \
    --batch_size $BATCH_SIZE \
    --seed $RANDOM \
    --coordconv \
    exp/cap/"$(basename $DATA_DIR)"_"$BACKBONE""_cc"
