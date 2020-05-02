#!/bin/bash

N_WORKERS="4"
N="1"

CL_CMD="cl run :lsl_shapeworld :shapeworld_4k"
CL_OPTS="--request-gpus 1 --request-cpus $N_WORKERS --request-memory 16g"

cl work lsl_acl20

# ==== MAIN ====
THIS_CMD="python3 lsl_shapeworld/train.py --cuda --batch_size 100 --seed 0 ."
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'Meta'"

THIS_CMD="python3 lsl_shapeworld/train.py --cuda --batch_size 100 --seed 1 . --predict_concept_hyp --hypo_lambda 20"
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'LSL'"

THIS_CMD="python3 lsl_shapeworld/train.py --cuda --batch_size 100 --seed 2 . --infer_hyp --hypo_lambda 1.0"
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'L3'"

# ==== LANGUAGE ABLATION ====
THIS_CMD="python3 lsl_shapeworld/train.py --cuda --predict_concept_hyp --hypo_lambda 20.0 --seed 3 --batch_size 100 --language_filter color ."
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'LSLColor'"

THIS_CMD="python3 lsl_shapeworld/train.py --cuda --predict_concept_hyp --hypo_lambda 20.0 --seed 4 --batch_size 100 --language_filter nocolor ."
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'LSLNoColor'"

THIS_CMD="python3 lsl_shapeworld/train.py --cuda --predict_concept_hyp --hypo_lambda 20.0 --seed 5 --batch_size 100 --shuffle_words ."
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'LSLShuffledWords'"

THIS_CMD="python3 lsl_shapeworld/train.py --cuda --predict_concept_hyp --hypo_lambda 20.0 --seed 6 --batch_size 100 --shuffle_captions ."
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'LSLShuffledCaptions'"

# ==== RESNET18 ====
THIS_CMD="python3 lsl_shapeworld/train.py --cuda --backbone resnet18 --batch_size 32 --seed 7 ."
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'MetaRN18'"

THIS_CMD="python3 lsl_shapeworld/train.py --cuda --backbone resnet18 --batch_size 32 --seed 8 . --predict_concept_hyp --hypo_lambda 20"
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'LSLRN18'"

THIS_CMD="python3 lsl_shapeworld/train.py --cuda --backbone resnet18 --batch_size 32 --seed 9 . --infer_hyp --hypo_lambda 1.0"
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'L3RN18'"

# ==== CONV4 ====
THIS_CMD="python3 lsl_shapeworld/train.py --cuda --backbone conv4 --batch_size 100 --seed 10 ."
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'MetaConv4'"

THIS_CMD="python3 lsl_shapeworld/train.py --cuda --backbone conv4 --batch_size 100 --seed 11 . --predict_concept_hyp --hypo_lambda 20"
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'LSLConv4'"

THIS_CMD="python3 lsl_shapeworld/train.py --cuda --backbone conv4 --batch_size 100 --seed 12 . --infer_hyp --hypo_lambda 1.0"
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'L3Conv4'"
