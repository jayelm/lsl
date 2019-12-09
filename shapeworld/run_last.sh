#!/bin/bash

# Meta
nlprun -g 1 -r 16G -c 4 -a py37-muj "
python lsl/train.py --cuda \
    --backbone conv4 \
    --batch_size 100 \
    --seed $RANDOM \
    --data_dir ~/Git/shapeworld_mine_10shot_easy \
    exp/last/meta_conv4
"

# Lambda @ 1
nlprun -g 1 -r 16G -c 4 -a py37-muj "
python lsl/train.py --cuda \
    --backbone conv4 \
    --predict_concept_hyp \
    --batch_size 100 \
    --hypo_lambda 1 \
    --seed $RANDOM \
    --data_dir ~/Git/shapeworld_mine_10shot_easy \
    exp/last/lsl_1_conv4
"

# Lambda @ 5
nlprun -g 1 -r 16G -c 4 -a py37-muj "
python lsl/train.py --cuda \
    --backbone conv4 \
    --predict_concept_hyp \
    --batch_size 100 \
    --hypo_lambda 5 \
    --seed $RANDOM \
    --data_dir ~/Git/shapeworld_mine_10shot_easy \
    exp/last/lsl_5_conv4
"

# Lambda @ 15
nlprun -g 1 -r 16G -c 4 -a py37-muj "
python lsl/train.py --cuda \
    --backbone conv4 \
    --predict_concept_hyp \
    --batch_size 100 \
    --hypo_lambda 15 \
    --seed $RANDOM \
    --data_dir ~/Git/shapeworld_mine_10shot_easy \
    exp/last/lsl_15_conv4
"

# L3
nlprun -g 1 -r 16G -c 4 -a py37-muj "
python lsl/train.py --cuda \
    --backbone conv4 \
    --infer_hyp \
    --batch_size 100 \
    --seed $RANDOM \
    --data_dir ~/Git/shapeworld_mine_10shot_easy \
    exp/last/l3_conv4
"
