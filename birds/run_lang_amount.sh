#!/bin/bash

for max_lang_per_class in 1 5 10 20 30 40 50 60; do
    # LSL
    python fewshot/run_cl.py --n 1 --log_dir exp/acl/language_amount/lsl_max_lang_$max_lang_per_class --lsl --glove_init --lang_lambda 5 --max_lang_per_class $max_lang_per_class --sample_class_lang

    # L3
    python fewshot/run_cl.py --n 1 --log_dir exp/acl/language_amount/l3_max_lang_$max_lang_per_class --l3 --glove_init --max_lang_per_class $max_lang_per_class --sample_class_lang
done
