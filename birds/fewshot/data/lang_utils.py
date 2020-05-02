"""
Utilities for processing language datasets
"""

import os
from collections import defaultdict

import numpy as np
import torch
from numpy import random
import torchfile

SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<END>"

COLOR_WORDS = set(
    [
        "amaranth",
        "charcoal",
        "amber",
        "amethyst",
        "apricot",
        "aquamarine",
        "azure",
        "baby blue",
        "beige",
        "black",
        "blue",
        "blush",
        "bronze",
        "brown",
        "burgundy",
        "byzantium",
        "carmine",
        "cerise",
        "cerulean",
        "champagne",
        "chartreuse",
        "chocolate",
        "cobalt",
        "coffee",
        "copper",
        "coral",
        "crimson",
        "cyan",
        "desert",
        "electric",
        "emerald",
        "erin",
        "gold",
        "gray",
        "grey",
        "green",
        "harlequin",
        "indigo",
        "ivory",
        "jade",
        "jungle",
        "lavender",
        "lemon",
        "lilac",
        "lime",
        "magenta",
        "magenta",
        "maroon",
        "mauve",
        "navy",
        "ochre",
        "olive",
        "orange",
        "orange",
        "orchid",
        "peach",
        "pear",
        "periwinkle",
        "persian",
        "pink",
        "plum",
        "prussian",
        "puce",
        "purple",
        "raspberry",
        "red",
        "red",
        "rose",
        "ruby",
        "salmon",
        "sangria",
        "sapphire",
        "scarlet",
        "silver",
        "slate",
        "spring",
        "spring",
        "tan",
        "taupe",
        "teal",
        "turquoise",
        "ultramarine",
        "violet",
        "viridian",
        "white",
        "yellow",
        "reddish",
        "yellowish",
        "greenish",
        "orangeish",
        "orangish",
        "blackish",
        "pinkish",
        "dark",
        "light",
        "bright",
        "greyish",
        "grayish",
        "brownish",
        "beigish",
        "aqua",
    ]
)


def filter_language(lang_tensor, language_filter, vocab):
    """
    Filter language, keeping or discarding color words

    :param lang_tensor: torch.Tensor of shape (n_imgs, lang_per_img,
        max_lang_len); language to be filtered
    :param language_filter: either 'color' or 'nocolor'; what language to
        filter out
    :param vocab: the vocabulary (so we know what indexes to remove)

    :returns: torch.Tensor of same shape as `lang_tensor` with either color or
        non-color words removed
    """
    assert language_filter in ["color", "nocolor"]

    cw = set(vocab[cw] for cw in COLOR_WORDS if cw in vocab)

    new_lang_tensor = torch.ones_like(lang_tensor)
    for bird_caps_i in range(lang_tensor.shape[0]):
        bird_caps = lang_tensor[bird_caps_i]
        new_bird_caps = torch.ones_like(bird_caps)
        for bird_cap_i in range(bird_caps.shape[0]):
            bird_cap = bird_caps[bird_cap_i]
            new_bird_cap = torch.ones_like(bird_cap)
            new_w_i = 0
            for w in bird_cap:
                is_cw = w.item() in cw
                if (language_filter == "color" and is_cw) or (
                    language_filter == "nocolor" and not is_cw
                ):
                    new_bird_cap[new_w_i] = w
                    new_w_i += 1
            if new_bird_cap[0].item() == 1:
                # FIXME: Here we're just choosing an arbitrary randomly
                # mispelled token; make a proper UNK token.
                new_bird_cap[0] = 5724
            new_bird_caps[bird_cap_i] = new_bird_cap
        new_lang_tensor[bird_caps_i] = new_bird_caps
    return new_lang_tensor


def shuffle_language(lang_tensor):
    """
    Scramble words in language

    :param lang_tensor: torch.Tensor of shape (n_img, lang_per_img, max_lang_len)

    :returns: torch.Tensor of same shape, but with words randomly scrambled
    """
    new_lang_tensor = torch.ones_like(lang_tensor)
    for bird_caps_i in range(lang_tensor.shape[0]):
        bird_caps = lang_tensor[bird_caps_i]
        new_bird_caps = torch.ones_like(bird_caps)
        for bird_cap_i in range(bird_caps.shape[0]):
            bird_cap = bird_caps[bird_cap_i]
            new_bird_cap = torch.ones_like(bird_cap)
            bird_cap_list = []
            for w in bird_cap.numpy():
                if w != 1:
                    bird_cap_list.append(w)
                else:
                    break
            random.shuffle(bird_cap_list)
            bird_cap_shuf = torch.tensor(
                bird_cap_list, dtype=new_bird_cap.dtype, requires_grad=False
            )
            new_bird_cap[: len(bird_cap_list)] = bird_cap_shuf
            new_bird_caps[bird_cap_i] = new_bird_cap
        new_lang_tensor[bird_caps_i] = new_bird_caps
    return new_lang_tensor


def get_lang_lengths(lang_tensor):
    """
    Get lengths of each caption

    :param lang_tensor: torch.Tensor of shape (n_img, lang_per_img, max_len)
    :returns: torch.Tensor of shape (n_img, lang_per_img)
    """
    max_lang_len = lang_tensor.shape[2]
    n_pad = torch.sum(lang_tensor == 0, dim=2)
    lang_lengths = max_lang_len - n_pad
    return lang_lengths


def get_lang_masks(lang_lengths, max_len=32):
    """
    Given lang lengths, convert to masks

    :param lang_lengths: torch.tensor of shape (n_imgs, lang_per_img)

    returns: torch.BoolTensor of shape (n_imgs, lang_per_img, max_len), binary
        mask with 0s in token spots and 1s in padding spots
    """
    mask = torch.ones(lang_lengths.shape + (max_len,), dtype=torch.bool)
    for i in range(lang_lengths.shape[0]):
        for j in range(lang_lengths.shape[1]):
            this_ll = lang_lengths[i, j]
            mask[i, j, :this_ll] = 0
    return mask


def add_sos_eos(lang_tensor, lang_lengths, vocab):
    """
    Pad language tensors

    :param lang: torch.Tensor of shape (n_imgs, n_lang_per_img, max_len)
    :param lang_lengths: torch.Tensor of shape (n_imgs, n_lang_per_img)
    :param vocab: dictionary from words -> idxs

    :returns: (lang, lang_lengths) where lang has SOS and EOS tokens added, and
        lang_lengths have all been increased by 2 (to account for SOS/EOS)
    """
    sos_idx = vocab[SOS_TOKEN]
    eos_idx = vocab[EOS_TOKEN]
    lang_tensor_padded = torch.zeros(
        lang_tensor.shape[0],
        lang_tensor.shape[1],
        lang_tensor.shape[2] + 2,
        dtype=torch.int64,
    )
    lang_tensor_padded[:, :, 0] = sos_idx
    lang_tensor_padded[:, :, 1:-1] = lang_tensor
    for i in range(lang_tensor_padded.shape[0]):
        for j in range(lang_tensor_padded.shape[1]):
            ll = lang_lengths[i, j]
            lang_tensor_padded[
                i, j, ll + 1
            ] = eos_idx  # + 1 accounts for sos token already there
    return lang_tensor_padded, lang_lengths + 2


def shuffle_lang_class(lang, lang_length, lang_mask):
    """
    For each class, shuffle captions across images

    :param lang: dict from class -> list of languages for that class
    :param lang_length: dict from class -> list of language lengths for that class
    :param lang_mask: list of language masks

    :returns: (new_lang, new_lang_length, new_lang_mask): tuple of new language
        dictionaries representing the modified language
    """
    new_lang = {}
    new_lang_length = {}
    new_lang_mask = {}
    for y in lang:
        # FIXME: Make this seedable
        img_range = np.arange(len(lang[y]))
        random.shuffle(img_range)
        nlang = []
        nlang_length = []
        nlang_mask = []
        for lang_i in img_range:
            nlang.append(lang[y][lang_i])
            nlang_length.append(lang_length[y][lang_i])
            nlang_mask.append(lang_mask[y][lang_i])
        new_lang[y] = nlang
        new_lang_length[y] = nlang_length
        new_lang_mask[y] = nlang_mask
    return new_lang, new_lang_length, new_lang_mask


def shuffle_all_class(lang, lang_length, lang_mask):
    """
    Shuffle captions completely randomly across all images and classes

    :param lang: dict from class -> list of languages for that class
    :param lang_length: dict from class -> list of language lengths for that class
    :param lang_mask: list of language masks

    :returns: (new_lang, new_lang_length, new_lang_mask): tuple of new language
        dictionaries representing the modified language
    """
    lens = [[(m, j) for j in range(len(lang[m]))] for m in lang.keys()]
    lens = [item for sublist in lens for item in sublist]
    shuffled_lens = lens[:]
    random.shuffle(shuffled_lens)
    new_lang = defaultdict(list)
    new_lang_length = defaultdict(list)
    new_lang_mask = defaultdict(list)
    for (m, _), (new_m, new_i) in zip(lens, shuffled_lens):
        new_lang[m].append(lang[new_m][new_i])
        new_lang_length[m].append(lang_length[new_m][new_i])
        new_lang_mask[m].append(lang_mask[new_m][new_i])
    assert all(len(new_lang[m]) == len(lang[m]) for m in lang.keys())
    return dict(new_lang), dict(new_lang_length), dict(new_lang_mask)


def load_vocab(lang_dir):
    """
    Load torch-serialized vocabulary from the lang dir

    :param: lang_dir: str, path to language directory
    :returns: dictionary from words -> idxs
    """
    vocab = torchfile.load(os.path.join(lang_dir, "vocab_c10.t7"))
    vocab = {k: v - 1 for k, v in vocab.items()}  # Decrement vocab
    vocab = {k.decode("utf-8"): v for k, v in vocab.items()}  # Unicode
    # Add SOS/EOS tokens
    sos_idx = len(vocab)
    vocab[SOS_TOKEN] = sos_idx
    eos_idx = len(vocab)
    vocab[EOS_TOKEN] = eos_idx
    return vocab


def glove_init(vocab, emb_size=300):
    """
    Initialize vocab with glove vectors. Requires spacy and en_vectors_web_lg
    spacy model

    :param vocab: dict from words -> idxs
    :param emb_size: int, size of embeddings (should be 300 for spacy glove
        vectors)

    :returns: torch.FloatTensor of size (len(vocab), emb_size), with glove
        embedding if exists, else zeros
    """
    import spacy

    try:
        nlp = spacy.load("en_vectors_web_lg", disable=["tagger", "parser", "ner"])
    except OSError:
        # Try loading for current directory (codalab)
        nlp = spacy.load(
            "./en_vectors_web_lg/en_vectors_web_lg-2.1.0/",
            disable=["tagger", "parser", "ner"],
        )

    vecs = np.zeros((len(vocab), emb_size), dtype=np.float32)
    vec_ids_sort = sorted(vocab.items(), key=lambda x: x[1])
    sos_idx = vocab[SOS_TOKEN]
    eos_idx = vocab[EOS_TOKEN]
    pad_idx = vocab[PAD_TOKEN]
    for vec, vecid in vec_ids_sort:
        if vecid in (pad_idx, sos_idx, eos_idx):
            v = np.zeros(emb_size, dtype=np.float32)
        else:
            v = nlp(vec)[0].vector
        vecs[vecid] = v
    vecs = torch.as_tensor(vecs)
    return vecs


def get_special_indices(vocab):
    """
    Get indices of special items from vocab.
    :param vocab: dictionary from words -> idxs
    :returns: dictionary from {sos_index, eos_index, pad_index} -> tokens
    """
    return {
        name: vocab[token]
        for name, token in [
            ("sos_index", SOS_TOKEN),
            ("eos_index", EOS_TOKEN),
            ("pad_index", PAD_TOKEN),
        ]
    }


def recycle_lang(langs, max_lang):
    """
    Given a limited amount of language, reuse `max_lang` times
    :param langs: list of languages
    :param max_lang: how long the full language tensor should be

    :returns: new_langs, a list of length `max_lang` created by cycling through
        `langs`
    """
    new_langs = []
    for i in range(len(langs)):
        new_langs.append(langs[i % max_lang])
    return new_langs
