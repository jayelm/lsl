#!/bin/bash

# Color
python fewshot/run_cl.py --n 1 --log_dir exp/acl/language_ablation/lsl_color --lsl --glove_init --lang_lambda 5 --language_filter color --max_lang_per_class 20 --sample_class_lang

# Nocolor
python fewshot/run_cl.py --n 1 --log_dir exp/acl/language_ablation/lsl_nocolor --lsl --glove_init --lang_lambda 5 --language_filter nocolor --max_lang_per_class 20 --sample_class_lang

# Shuffled words
python fewshot/run_cl.py --n 1 --log_dir exp/acl/language_ablation/lsl_shuffled_words --lsl --glove_init --lang_lambda 5 --shuffle_lang --max_lang_per_class 20 --sample_class_lang

# Shuffled captions
python fewshot/run_cl.py --n 1 --log_dir exp/acl/language_ablation/lsl_shuffled_captions --lsl --glove_init --lang_lambda 5 --scramble_all --max_lang_per_class 20 --sample_class_lang
