"""
Dataset utilities
"""

import os
import json
import logging

import torch
import numpy as np
import torch.utils.data as data
from torchvision import transforms

from utils import next_random, OrderedCounter

# Set your data directory here!
DATA_DIR = '/home/songlin/'
SPLIT_OPTIONS = ['train', 'val', 'test', 'val_same', 'test_same']

logging.getLogger(__name__).setLevel(logging.INFO)

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
N_EX = 4  # number of examples per task

random = next_random()
COLORS = {
    'black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white'
}
SHAPES = {
    'square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle',
    'semicircle', 'ellipse'
}


class ClassNoise:
    def __init__(self, n, max_rgb=0.1):
        self.n = n
        self.max_rgb = max_rgb
        self.dict = {}
        self.colors = [
            torch.zeros(3).uniform_(0, self.max_rgb) for i in range(n)
        ]
        self.curr_n = 0

    def __getitem__(self, i):
        if i not in self.dict:
            self.dict[i] = self.colors[self.curr_n]
            self.curr_n = (self.curr_n + 1) % self.n
        return self.dict[i]

    def __contains__(self, i):
        return i in self.dict

    def __len__(self):
        return len(self.dict)

    def random_noise(self, n):
        if n == 1:
            return torch.zeros(3).uniform_(0, self.max_rgb)
        else:
            noise = torch.zeros(n * 3).uniform_(0, self.max_rgb)
            return noise.view(n, 3)


def gen_class_noise(image_dim, noise, noise_type='gaussian'):
    if noise_type == 'gaussian':
        if noise != 0.0:
            return torch.zeros(image_dim,
                               requires_grad=False).normal_(0, noise)
        else:
            return torch.zeros(image_dim, requires_grad=False)
    else:
        return torch.zeros(image_dim,
                           requires_grad=False).uniform_(-noise, noise)


def get_max_hint_length(data_dir=None):
    """
    Get the maximum number of words in a sentence across all splits
    """
    if data_dir is None:
        data_dir = DATA_DIR
    max_len = 0
    for split in ['train', 'val', 'test', 'val_same', 'test_same']:
        for tf in ['hints.json', 'test_hints.json']:
            hints_file = os.path.join(data_dir, 'shapeworld', split, tf)
            if os.path.exists(hints_file):
                with open(hints_file) as fp:
                    hints = json.load(fp)
                split_max_len = max([len(hint.split()) for hint in hints])
                if split_max_len > max_len:
                    max_len = split_max_len
    if max_len == 0:
        raise RuntimeError("Can't find any splits in {}".format(data_dir))
    return max_len


def get_black_mask(imgs):
    if len(imgs.shape) == 4:
        # Then color is 1st dim
        col_dim = 1
    else:
        col_dim = 0
    total = imgs.sum(dim=col_dim)

    # Put dim back in
    is_black = total == 0.0
    is_black = is_black.unsqueeze(col_dim).expand_as(imgs)

    return is_black


class ShapeWorld(data.Dataset):
    r"""Loader for ShapeWorld data as in L3.

    @param split: string [default: train]
                  train|val|test|val_same|test_same
    @param vocab: ?Object [default: None]
                  initialize with a vocabulary
                  important to do this for validation/test set.
    @param augment: boolean [default: False]
                    negatively sample data from other concepts.
    @param max_size: limit size to this many training examples
    @param precomputed_features: load precomputed VGG features rather than raw image data
    @param noise: amount of uniform noise to add to examples
    @param class_noise_weight: how much of the noise added to examples should
                               be the same across (pos/neg classes) (between
                               0.0 and 1.0)

    NOTE: for now noise/class_noise_weight has no impact on val/test datasets
    """

    def __init__(self,
                 split='train',
                 vocab=None,
                 augment=False,
                 max_size=None,
                 precomputed_features=True,
                 preprocess=False,
                 noise=0.0,
                 class_noise_weight=0.5,
                 fixed_noise_colors=None,
                 fixed_noise_colors_max_rgb=0.2,
                 noise_type='gaussian',
                 data_dir=None,
                 language_filter=None,
                 shuffle_words=False,
                 shuffle_captions=False):
        super(ShapeWorld, self).__init__()
        self.split = split
        assert self.split in SPLIT_OPTIONS
        self.vocab = vocab
        self.augment = augment
        self.max_size = max_size

        assert noise_type in ('gaussian', 'normal')
        self.noise_type = noise_type

        # Positive class noise
        if precomputed_features:
            self.image_dim = (4608, )
        else:
            self.image_dim = (3, 64, 64)

        self.noise = noise
        self.fixed_noise_colors = fixed_noise_colors
        self.fixed_noise_colors_max_rgb = fixed_noise_colors_max_rgb
        if not class_noise_weight >= 0.0 and class_noise_weight <= 1.0:
            raise ValueError(
                "Class noise weight must be between 0 and 1, got {}".format(
                    class_noise_weight))
        self.class_noise_weight = class_noise_weight

        if self.fixed_noise_colors is not None:
            self.class_noises = ClassNoise(
                self.fixed_noise_colors,
                max_rgb=self.fixed_noise_colors_max_rgb)
        else:
            self.class_noises = {
                1: gen_class_noise(self.image_dim, self.noise,
                                   self.noise_type),
                0: gen_class_noise(self.image_dim, self.noise, self.noise_type)
            }

        if data_dir is None:
            data_dir = DATA_DIR
        self.data_dir = data_dir
        split_dir = os.path.join(data_dir, 'shapeworld', split)
        if not os.path.exists(split_dir):
            raise RuntimeError("Can't find {}".format(split_dir))

        self.precomputed_features = precomputed_features
        if self.precomputed_features:
            in_features_name = 'inputs.feats.npz'
            ex_features_name = 'examples.feats.npz'
        else:
            in_features_name = 'inputs.npz'
            ex_features_name = 'examples.npz'

        self.preprocess = None
        if preprocess:
            self.preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        # hints = language
        # examples = images with positive labels (pre-training)
        # input = test time input
        # label = test time label
        labels = np.load(os.path.join(split_dir, 'labels.npz'))['arr_0']
        in_features = np.load(os.path.join(split_dir, in_features_name))['arr_0']
        ex_features = np.load(os.path.join(split_dir, ex_features_name))['arr_0']
        with open(os.path.join(split_dir, 'hints.json')) as fp:
            hints = json.load(fp)

        test_hints = os.path.join(split_dir, 'test_hints.json')
        if self.fixed_noise_colors is not None:
            assert os.path.exists(test_hints)
        if os.path.exists(test_hints):
            with open(test_hints, 'r') as fp:
                test_hints = json.load(fp)
            self.test_hints = test_hints
        else:
            self.test_hints = None

        if self.test_hints is not None:
            for a, b, label in zip(hints, test_hints, labels):
                if label:
                    assert a == b, (a, b, label)
                #  else:  # XXX: What?/
                #  assert a != b, (a, b, label)

        if not self.precomputed_features:
            # Bring channel to first dim
            in_features = np.transpose(in_features, (0, 3, 1, 2))
            ex_features = np.transpose(ex_features, (0, 1, 4, 2, 3))

        if self.max_size is not None:
            labels = labels[:self.max_size]
            in_features = in_features[:self.max_size]
            ex_features = ex_features[:self.max_size]
            hints = hints[:self.max_size]

        n_data = len(hints)

        self.in_features = in_features
        self.ex_features = ex_features
        self.hints = hints

        if self.vocab is None:
            self.create_vocab(hints, test_hints)

        self.w2i, self.i2w = self.vocab['w2i'], self.vocab['i2w']
        self.vocab_size = len(self.w2i)

        # Language processing
        self.language_filter = language_filter
        if self.language_filter is not None:
            assert self.language_filter in ['color', 'nocolor']
        self.shuffle_words = shuffle_words
        self.shuffle_captions = shuffle_captions

        # this is the maximum number of tokens in a sentence
        max_length = get_max_hint_length(data_dir)

        hints, hint_lengths = [], []
        for hint in self.hints:
            hint_tokens = hint.split()
            # Hint processing
            if self.language_filter == 'color':
                hint_tokens = [t for t in hint_tokens if t in COLORS]
            elif self.language_filter == 'nocolor':
                hint_tokens = [t for t in hint_tokens if t not in COLORS]
            if self.shuffle_words:
                random.shuffle(hint_tokens)

            hint = [SOS_TOKEN, *hint_tokens, EOS_TOKEN]
            hint_length = len(hint)

            hint.extend([PAD_TOKEN] * (max_length + 2 - hint_length))
            hint = [self.w2i.get(w, self.w2i[UNK_TOKEN]) for w in hint]

            hints.append(hint)
            hint_lengths.append(hint_length)

        hints = np.array(hints)
        hint_lengths = np.array(hint_lengths)

        if self.test_hints is not None:
            test_hints, test_hint_lengths = [], []
            for test_hint in self.test_hints:
                test_hint_tokens = test_hint.split()

                if self.language_filter == 'color':
                    test_hint_tokens = [
                        t for t in test_hint_tokens if t in COLORS
                    ]
                elif self.language_filter == 'nocolor':
                    test_hint_tokens = [
                        t for t in test_hint_tokens if t not in COLORS
                    ]
                if self.shuffle_words:
                    random.shuffle(test_hint_tokens)

                test_hint = [SOS_TOKEN, *test_hint_tokens, EOS_TOKEN]
                test_hint_length = len(test_hint)

                test_hint.extend([PAD_TOKEN] * (max_length + 2 - test_hint_length))

                test_hint = [
                    self.w2i.get(w, self.w2i[UNK_TOKEN]) for w in test_hint
                ]

                test_hints.append(test_hint)
                test_hint_lengths.append(test_hint_length)

            test_hints = np.array(test_hints)
            test_hint_lengths = np.array(test_hint_lengths)

        data = []
        for i in range(n_data):
            if self.shuffle_captions:
                hint_i = random.randint(len(hints))
                test_hint_i = random.randint(len(test_hints))
            else:
                hint_i = i
                test_hint_i = i
            if self.test_hints is not None:
                th = test_hints[test_hint_i]
                thl = test_hint_lengths[test_hint_i]
            else:
                th = hints[test_hint_i]
                thl = hint_lengths[test_hint_i]
            data_i = (ex_features[i], in_features[i], labels[i], hints[hint_i],
                      hint_lengths[hint_i], th, thl)
            data.append(data_i)

        self.data = data
        self.max_length = max_length


    def create_vocab(self, hints, test_hints):
        w2i = dict()
        i2w = dict()
        w2c = OrderedCounter()

        special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        for hint in hints:
            hint_tokens = hint.split()
            w2c.update(hint_tokens)

        if test_hints is not None:
            for hint in test_hints:
                hint_tokens = hint.split()
                w2c.update(hint_tokens)

        for w, c in list(w2c.items()):
            i2w[len(w2i)] = w
            w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)
        vocab = dict(w2i=w2i, i2w=i2w)
        self.vocab = vocab

        logging.info('Created vocab with %d words.' % len(w2c))

    def __len__(self):
        return len(self.data)

    def sample_train(self, n_batch):
        assert self.split == 'train'
        n_train = len(self.data)
        batch_examples = []
        batch_image = []
        batch_label = []
        batch_hint = []
        batch_hint_length = []
        if self.test_hints is not None:
            batch_test_hint = []
            batch_test_hint_length = []

        for _ in range(n_batch):
            index = random.randint(n_train)
            examples, image, label, hint, hint_length, test_hint, test_hint_length = \
                self.__getitem__(index)

            batch_examples.append(examples)
            batch_image.append(image)
            batch_label.append(label)
            batch_hint.append(hint)
            batch_hint_length.append(hint_length)
            if self.test_hints is not None:
                batch_test_hint.append(test_hint)
                batch_test_hint_length.append(test_hint_length)

        batch_examples = torch.stack(batch_examples)
        batch_image = torch.stack(batch_image)
        batch_label = torch.from_numpy(np.array(batch_label)).long()
        batch_hint = torch.stack(batch_hint)
        batch_hint_length = torch.from_numpy(
            np.array(batch_hint_length)).long()
        if self.test_hints is not None:
            batch_test_hint = torch.stack(batch_test_hint)
            batch_test_hint_length = torch.from_numpy(
                np.array(batch_test_hint_length)).long()
        else:
            batch_test_hint = None
            batch_test_hint_length = None

        return (
            batch_examples, batch_image, batch_label, batch_hint,
            batch_hint_length, batch_test_hint, batch_test_hint_length
        )

    def add_fixed_noise_colors(self,
                               support,
                               query,
                               support_hint,
                               query_hint,
                               clamp=False):
        # Get hashable version of support/query hint
        support_hash = tuple(support_hint.numpy().tolist())
        query_hash = tuple(query_hint.numpy().tolist())
        # Identify black parts of background image
        # Class noise depends on support and query hints
        support_class_noise = self.class_noises[support_hash]
        query_class_noise = self.class_noises[query_hash]

        # Random noise for instances
        support_instance_noise = self.class_noises.random_noise(
            support.shape[0])
        query_instance_noise = self.class_noises.random_noise(1)

        # Add dimensions to noise
        support_class_noise = support_class_noise.unsqueeze(0).unsqueeze(
            2).unsqueeze(3).expand_as(support)
        query_class_noise = query_class_noise.unsqueeze(1).unsqueeze(
            2).expand_as(query)
        # BG color fixed for each image, so no need to unsqueeze along image dim
        support_instance_noise = support_instance_noise.unsqueeze(2).unsqueeze(
            3).expand_as(support)
        query_instance_noise = query_instance_noise.unsqueeze(1).unsqueeze(
            2).expand_as(query)

        # Get mask of black pixels
        support_mask = get_black_mask(support).float()
        query_mask = get_black_mask(query).float()

        # Combine class vs query noise
        support_noise = (self.class_noise_weight * support_class_noise) + (
            (1 - self.class_noise_weight) * support_instance_noise)
        query_noise = (self.class_noise_weight * query_class_noise) + (
            (1 - self.class_noise_weight) * query_instance_noise)

        # Fill in black pixels with random color
        support_noised = (
            (1 - support_mask) * support) + (support_mask * support_noise)
        query_noised = ((1 - query_mask) * query) + (query_mask * query_noise)

        return support_noised, query_noised

    def add_noise(self, support, query, y, clamp=False):
        # Fixed pos/neg class noise
        support_class_noise = self.class_noises[1]
        query_class_noise = self.class_noises[y]

        return self._add_noise(support,
                               query,
                               support_class_noise,
                               query_class_noise,
                               clamp=clamp)

    def _add_noise(self,
                   support,
                   query,
                   support_class_noise,
                   query_class_noise,
                   clamp=False):
        # Expand to match examples
        support_class_noise = support_class_noise.unsqueeze(0).expand_as(
            support)
        # Instance noise
        if self.noise == 0:
            support_instance_noise = torch.zeros(*support.shape, dtype=torch.float)
            query_instance_noise = torch.zeros(*query.shape, dtype=torch.float)
        elif self.noise_type == 'gaussian':
            support_instance_noise = torch.FloatTensor(*support.shape).normal_(
                0, self.noise)
            query_instance_noise = torch.FloatTensor(*query.shape).normal_(
                0, self.noise)
        else:
            support_instance_noise = torch.FloatTensor(
                *support.shape).uniform_(-self.noise, self.noise)
            query_instance_noise = torch.FloatTensor(*query.shape).uniform_(
                -self.noise, self.noise)

        # Compute weighted noise
        query_noise = (self.class_noise_weight * query_class_noise) + (
            (1 - self.class_noise_weight) * query_instance_noise)
        support_noise = (self.class_noise_weight * support_class_noise) + (
            (1 - self.class_noise_weight) * support_instance_noise)

        support_noised = support + support_noise
        query_noised = query + query_noise

        # Add to expands
        if clamp:
            support_noised = torch.clamp(support_noised, 0.0, 1.0)
            query_noised = torch.clamp(query_noised, 0.0, 1.0)

        return support_noised, query_noised

    def __getitem__(self, index):
        if self.split == 'train' and self.augment:
            examples, image, label, hint, hint_length, test_hint, test_hint_length = self.data[
                index]

            # tie a language to a concept; convert to pytorch.
            hint = torch.from_numpy(hint).long()
            test_hint = torch.from_numpy(test_hint).long()
            examples = torch.clone(torch.from_numpy(examples)).float()

            # in training, pick whether to show positive or negative example.
            sample_label = random.randint(2)
            n_train = len(self.data)

            if sample_label == 0:
                # if we are training, we need to negatively sample data and
                # return a tuple (example_z, hint_z, 1) or...
                # return a tuple (example_z, hint_other_z, 0).
                # Sample a new test hint as well.
                examples2, image2, _, support_hint2, support_hint_length2, query_hint2, query_hint_length2 = self.data[
                    random.randint(n_train)]

                # pick either an example or an image as the query image for negative samples.
                swap = random.randint(N_EX + 1)
                if swap == N_EX:
                    feats = image2
                    # Use the QUERY hint of the new example
                    test_hint = query_hint2
                    test_hint_length = query_hint_length2
                else:
                    feats = examples2[swap, ...]
                    # Use the SUPPORT hint of the new example
                    test_hint = support_hint2
                    test_hint_length = support_hint_length2

                test_hint = torch.from_numpy(test_hint).long()

                feats = torch.from_numpy(feats).float()
                # this is a 0 since feats does not match this hint.
                if self.fixed_noise_colors is not None:
                    examples, feats = self.add_fixed_noise_colors(
                        examples,
                        feats,
                        hint,
                        test_hint,
                        clamp=not self.precomputed_features)
                else:
                    examples, feats = self.add_noise(
                        examples,
                        feats,
                        0,
                        clamp=not self.precomputed_features)
                if self.preprocess is not None:
                    feats = self.preprocess(feats)
                    examples = torch.stack(
                        [self.preprocess(e) for e in examples])
                return examples, feats, 0, hint, hint_length, test_hint, test_hint_length
            else:  # sample_label == 1
                swap = random.randint((N_EX + 1 if label == 1 else N_EX))
                # pick either an example or an image.
                
                image = torch.clone(torch.from_numpy(image))

                if swap == N_EX:
                    feats = image
                else:
                    # if we need to swap image with one of examples
                    feats = examples[swap, ...]
                    if label == 1:
                        examples[swap, ...] = image
                    else:
                        # duplicate an image from examples if image and examples belong to different concepts
                        examples[swap, ...] = examples[random.randint(N_EX
                                                                      ), ...]

                # This is a positive example, so whatever example we've chosen,
                # assume the query hint matches the support hint.
                test_hint = hint
                test_hint_length = hint_length

                feats = feats.float()

                if self.fixed_noise_colors is not None:
                    examples, feats = self.add_fixed_noise_colors(
                        examples,
                        feats,
                        hint,
                        test_hint,
                        clamp=not self.precomputed_features)
                else:
                    examples, feats = self.add_noise(
                        examples,
                        feats,
                        1,
                        clamp=not self.precomputed_features)

                if self.preprocess is not None:
                    feats = self.preprocess(feats)
                    examples = torch.stack(
                        [self.preprocess(e) for e in examples])
                return examples, feats, 1, hint, hint_length, test_hint, test_hint_length

        else:  # val, val_same, test, test_same
            examples, image, label, hint, hint_length, test_hint, test_hint_length = self.data[
                index]

            # no fancy stuff. just return image.
            image = torch.from_numpy(image).float()

            # NOTE: we provide the oracle text.
            hint = torch.from_numpy(hint).long()
            test_hint = torch.from_numpy(test_hint).long()
            examples = torch.from_numpy(examples).float()

            # this is a 0 since feats does not match this hint.
            if self.fixed_noise_colors is not None:
                examples, image = self.add_fixed_noise_colors(
                    examples,
                    image,
                    hint,
                    test_hint,
                    clamp=not self.precomputed_features)
            else:
                examples, image = self.add_noise(
                    examples, image, 0, clamp=not self.precomputed_features)

            if self.preprocess is not None:
                image = self.preprocess(image)
                examples = torch.stack([self.preprocess(e) for e in examples])

            return examples, image, label, hint, hint_length, test_hint, test_hint_length

    def to_text(self, hints):
        texts = []
        for hint in hints:
            text = []
            for tok in hint:
                i = tok.item()
                w = self.vocab['i2w'].get(i, UNK_TOKEN)
                if w == PAD_TOKEN:
                    break
                text.append(w)
            texts.append(text)

        return texts


def extract_features(hints):
    """
    Extract features from hints
    """
    all_feats = []
    for hint in hints:
        feats = []
        for maybe_rel in ['above', 'below', 'left', 'right']:
            if maybe_rel in hint:
                rel = maybe_rel
                rel_idx = hint.index(rel)
                break
        else:
            raise RuntimeError("Didn't find relation: {}".format(hint))
        # Add relation
        feats.append('rel:{}'.format(rel))
        fst, snd = hint[:rel_idx], hint[rel_idx:]
        # fst: [<sos>, a, ..., is]
        fst_shape = fst[2:fst.index('is')]
        # snd: [..., a, ..., ., <eos>]
        try:
            snd_shape = snd[snd.index('a') + 1:-2]
        except ValueError:
            # Use "an"
            snd_shape = snd[snd.index('an') + 1:-2]

        for name, fragment in [('fst', fst_shape), ('snd', snd_shape)]:
            for feat in fragment:
                if feat != 'shape':
                    if feat in COLORS:
                        feats.append('{}:color:{}'.format(name, feat))
                    else:
                        assert feat in SHAPES, hint
                        feats.append('{}:shape:{}'.format(name, feat))
        all_feats.append(feats)
    return all_feats
