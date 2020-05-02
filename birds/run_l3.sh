#!/bin/bash

python fewshot/run_cl.py --n 1 --log_dir exp/acl/l3 --l3 --glove_init --lang_lambda 5 --max_lang_per_class 20 --sample_class_lang
