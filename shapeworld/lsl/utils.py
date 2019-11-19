"""
Utilities
"""

from collections import Counter, OrderedDict
import json
import os
import shutil

import numpy as np
import torch

random_counter = [0]


def next_random():
    random = np.random.RandomState(random_counter[0])
    random_counter[0] += 1
    return random


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self), )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, raw=False):
        self.raw = raw
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.raw:
            self.raw_scores = []

    def update(self, val, n=1, raw_scores=None):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.raw:
            self.raw_scores.extend(list(raw_scores))


def save_checkpoint(state, is_best, folder='./',
                    filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def merge_args_with_dict(args, dic):
    for k, v in list(dic.items()):
        setattr(args, k, v)


def make_output_and_sample_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sample_dir = os.path.join(out_dir, 'samples')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    return out_dir, sample_dir


def save_defaultdict_to_fs(d, out_path):
    d = dict(d)
    with open(out_path, 'w') as fp:
        d_str = json.dumps(d, ensure_ascii=True)
        fp.write(d_str)


def idx2word(idx, i2w):
    sent_str = [str()] * len(idx)
    for i, sent in enumerate(idx):
        for word_id in sent:
            sent_str[i] += str(i2w[word_id.item()]) + " "
        sent_str[i] = sent_str[i].strip()

    return sent_str
