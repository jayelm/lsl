#!/bin/bash

N_WORKERS="4"
N="1"

CL_CMD="cl run :filelists :fewshot :reed-birds :custom_filelists :en_vectors_web_lg"
CL_OPTS="--request-gpus 1 --request-cpus $N_WORKERS --request-memory 16g"

LANG_LAMBDA="5"
LANG_HIDDEN_SIZE="200"
MAX_LANG_PER_CLASS="20"

CMD="python3 fewshot/run_cl.py --n_workers $N_WORKERS --n $N --log_dir ./"

cl work lsl_acl20

# ==== MAIN ====
THIS_CMD="$CMD"
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'Meta'"

THIS_CMD="$CMD --lsl --glove_init --lang_lambda $LANG_LAMBDA --lang_hidden_size $LANG_HIDDEN_SIZE --max_lang_per_class $MAX_LANG_PER_CLASS --sample_class_lang"
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'LSL'"

THIS_CMD="$CMD --lsl --glove_init --lang_lambda $LANG_LAMBDA --lang_hidden_size $LANG_HIDDEN_SIZE --max_lang_per_class $MAX_LANG_PER_CLASS --sample_class_lang"
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'LSL'"

THIS_CMD="$CMD --lsl --glove_init --lang_lambda $LANG_LAMBDA --lang_hidden_size $LANG_HIDDEN_SIZE --max_lang_per_class $MAX_LANG_PER_CLASS --sample_class_lang"
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'LSL'"

THIS_CMD="$CMD --l3 --glove_init --lang_hidden_size $LANG_HIDDEN_SIZE --max_lang_per_class $MAX_LANG_PER_CLASS --sample_class_lang"
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'L3'"

# ==== LANGUAGE ABLATION ====
THIS_CMD="$CMD --lsl --glove_init --lang_lambda $LANG_LAMBDA --language_filter color --max_lang_per_class $MAX_LANG_PER_CLASS --sample_class_lang"
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'Color'"

THIS_CMD="$CMD --lsl --glove_init --lang_lambda $LANG_LAMBDA --language_filter nocolor --max_lang_per_class $MAX_LANG_PER_CLASS --sample_class_lang"
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'NoColor'"

THIS_CMD="$CMD --lsl --glove_init --lang_lambda $LANG_LAMBDA --shuffle_lang --max_lang_per_class $MAX_LANG_PER_CLASS --sample_class_lang"
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'ShuffledWords'"

THIS_CMD="$CMD --lsl --glove_init --lang_lambda $LANG_LAMBDA --scramble_all --max_lang_per_class $MAX_LANG_PER_CLASS --sample_class_lang"
eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'ShuffledCaptions'"

# ==== LANGUAGE AMOUNT ====
for MLPC in 1 5 10 20 30 40 50 60; do
    for i in 1 2 3; do
        # LSL
        THIS_CMD="python3 fewshot/run_cl.py --n_workers $N_WORKERS --n $N --log_dir ./ --lsl --glove_init --lang_lambda $LANG_LAMBDA --lang_hidden_size $LANG_HIDDEN_SIZE --max_lang_per_class $MLPC --sample_class_lang"
        eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'LSL-$MLPC-$i'"

        # L3
        THIS_CMD="python3 fewshot/run_cl.py --n_workers $N_WORKERS --n $N --log_dir ./ --l3 --glove_init --max_lang_per_class $MLPC --sample_class_lang"
        eval "$CL_CMD '$THIS_CMD' $CL_OPTS -n 'L3-$MLPC-$i'"
    done
done
