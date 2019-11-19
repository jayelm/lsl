#!/bin/bash

set -e

python lsl/train.py --cuda \
    --predict_concept_hyp \
    --hypo_lambda 20.0 \
    --seed "$RANDOM" \
    --batch_size 100 \
    --language_filter color \
    exp/lsl_color

python lsl/train.py --cuda \
    --predict_concept_hyp \
    --hypo_lambda 20.0 \
    --seed "$RANDOM" \
    --batch_size 100 \
    --language_filter nocolor \
    exp/lsl_nocolor

python lsl/train.py --cuda \
    --predict_concept_hyp \
    --hypo_lambda 20.0 \
    --seed "$RANDOM" \
    --batch_size 100 \
    --shuffle_words \
    exp/lsl_shuffle_words

python lsl/train.py --cuda \
    --predict_concept_hyp \
    --hypo_lambda 20.0 \
    --seed "$RANDOM" \
    --batch_size 100 \
    --shuffle_captions \
    exp/lsl_shuffle_captions
