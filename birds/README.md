# LSL - Birds

This codebase is built off of [wyharveychen/CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot) ([paper](https://openreview.net/pdf?id=HkxLXnAcFQ)) - thanks to them!

## Dependencies

Tested with Python 3.7.4, torch 1.4.0, torchvision 0.4.1, numpy 1.16.2, PIL
5.4.1, torchfile 0.1.0, sklearn 0.20.3, pandas 0.25.2

Glove initialization depends on spacy 2.2.2 and the spacy `en_vectors_web_lg`
model:

```
python -m spacy.download en_vectors_web_lg
```

## Data

To download data, cd to `filelists/CUB` and run `source download_CUB.sh`. This
downloads the CUB 200-2011 dataset and also runs `python write_CUB_filelist.py`.

`python write_CUB_filelist.py` saves a filelist (train/val/test) split
to `./custom_filelists/CUB/{base,val,novel}.json`.

Then run `python save_np.py` which takes the images and serializes them as NP arrays
(for speed).

The language data is available from
[reedscot/cvpr2016](https://github.com/reedscot/cvpr2016) ([GDrive link](https://drive.google.com/open?id=0B0ywwgffWnLLZW9uVHNjb2JmNlE)). Download it and unzip to `reed-birds` directory in the main directory (e.g. the path to the vocab file should be `./reed-birds/vocab_c10.t7`).

## Running

To train and evaluate a model, you will run `fewshot/train.py` and `fewshot/test.py`,
respectively. Alternatively, for CodaLab, the `fewshot/run_cl.py` script does
both training and testing, with slightly more friendly argument names
(`fewshot/run_cl.py --help`) for more.

The shell scripts contain commands for running the various models:

- `run_meta.sh`: Non-linguistic protonet baseline
- `run_l3.sh`: learning with latent language (Andreas et al., 2018)
- `run_lsl.sh`: Ours
- `run_lang_ablation.sh`: Language ablation studies
- `run_lang_amount.sh`: Language amount

## References

(from the original CloserLookFewShot repo)

Our testbed builds upon several existing publicly available code. Specifically, we have modified and integrated the following code into this project:

* Framework, Backbone, Method: Matching Network
https://github.com/facebookresearch/low-shot-shrink-hallucinate
* Omniglot dataset, Method: Prototypical Network
https://github.com/jakesnell/prototypical-networks
* Method: Relational Network
https://github.com/floodsung/LearningToCompare_FSL
* Method: MAML
https://github.com/cbfinn/maml
https://github.com/dragen1860/MAML-Pytorch
https://github.com/katerakelly/pytorch-maml
