# LSL - ShapeWorld experiments

This code is graciously adapted from code written by [Mike Wu](https://www.mikehwu.com/).

## Data

Download data [here](http://nlp.stanford.edu/data/muj/shapeworld_4k.zip)
(~1GB zipped, 12G unzipped). Unzip, and set `DATA_DIR` in `datasets.py` to be
point to the folder *containing* the ShapeWorld folder you just unzipped.

The code also works with Jacob Andreas' [original ShapeWorld data
files](http://people.eecs.berkeley.edu/~jda/data/shapeworld.tar.gz); results
are the same, but with higher variance on test accuracies.

For more details on the dataset (and how to reproduce it), check
[jacoobandreas/l3](https://github.com/jacobandreas/l3) and the accompanying
[paper](https://arxiv.org/abs/1711.00482)

## Running

The models can be run with the scripts in this directory:

- `run_l3.sh` - L3
- `run_lsl.sh` - LSL (ours)
- `run_lsl_img.sh` - LSL, but decoding captions from the image embeddings
    instead of the concept (not reported)
- `run_meta.sh` - meta-learning baseline
- `run_lang_ablation.sh` - language ablation studies

They will output results in the `exp/` directory (runs for ViGIL are already
present there)

## Analysis

`analysis/metrics.Rmd` contains `R` code for reproducing the plots in the
paper.
