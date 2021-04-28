#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -u lsl/train.py --cuda \
    --infer_hyp \
    --backbone resnet18\
    --hypo_lambda 1.0 \
    --batch_size 32 \
    --seed $RANDOM \
    --comparison dotp \
    --epochs 50 \
    --hint_retriever l2 \
    --plot_bleu_score \
    exp/l3/debug > l3_resnet_dopt_l2_full_g.out
    # --oracle \
