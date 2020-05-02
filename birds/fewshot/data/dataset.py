# This code is modified from
# https://github.com/facebookresearch/low-shot-shrink-hallucinate

import glob
import json
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from numpy import random
from PIL import Image
import torchfile

from . import lang_utils


CUB_IMAGES_PATH = "CUB_200_2011/images"


def identity(x):
    return x


def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return img


class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, "r") as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = os.path.join(self.meta["image_names"][i])
        img = load_image(image_path)
        img = self.transform(img)
        target = self.target_transform(self.meta["image_labels"][i])
        return img, target

    def __len__(self):
        return len(self.meta["image_names"])


class SetDataset:
    def __init__(
        self,
        name,
        data_file,
        batch_size,
        transform,
        args=None,
        lang_dir=None,
        vocab=None,
        max_class=None,
        max_img_per_class=None,
        max_lang_per_class=None,
    ):
        self.name = name
        with open(data_file, "r") as f:
            self.meta = json.load(f)

        self.args = args
        self.max_class = max_class
        self.max_img_per_class = max_img_per_class
        self.max_lang_per_class = max_lang_per_class

        if not (1 <= args.n_caption <= 10):
            raise ValueError("Invalid # captions {}".format(args.n_caption))

        self.cl_list = np.unique(self.meta["image_labels"]).tolist()

        if self.max_class is not None:
            if self.max_class > len(self.cl_list):
                raise ValueError(
                    "max_class set to {} but only {} classes in {}".format(
                        self.max_class, len(self.cl_list), data_file
                    )
                )
            self.cl_list = self.cl_list[: self.max_class]

        if args.language_filter not in ["all", "color", "nocolor"]:
            raise NotImplementedError(
                "language_filter = {}".format(args.language_filter)
            )

        self.sub_meta_lang = {}
        self.sub_meta_lang_length = {}
        self.sub_meta_lang_mask = {}
        self.sub_meta = {}

        for cl in self.cl_list:
            self.sub_meta[cl] = []
            self.sub_meta_lang[cl] = []
            self.sub_meta_lang_length[cl] = []
            self.sub_meta_lang_mask[cl] = []

        # Load language and mapping from image names -> lang idx
        self.lang = {}
        self.lang_lengths = {}
        self.lang_masks = {}
        self.image_name_idx = {}
        for cln, label_name in enumerate(self.meta["label_names"]):
            # Use the numeric class id instead of label name due to
            # inconsistencies
            digits = label_name.split(".")[0]
            matching_names = [
                x
                for x in os.listdir(os.path.join(lang_dir, "word_c10"))
                if x.startswith(digits)
            ]
            assert len(matching_names) == 1, matching_names
            label_file = os.path.join(lang_dir, "word_c10", matching_names[0])
            lang_tensor = torch.from_numpy(torchfile.load(label_file)).long()
            # Make words last dim
            lang_tensor = lang_tensor.transpose(2, 1)
            lang_tensor = lang_tensor - 1  # XXX: Decrement language by 1 upon load

            if (
                self.args.language_filter == "color"
                or self.args.language_filter == "nocolor"
            ):
                lang_tensor = lang_utils.filter_language(
                    lang_tensor, self.args.language_filter, vocab
                )

            if self.args.shuffle_lang:
                lang_tensor = lang_utils.shuffle_language(lang_tensor)

            lang_lengths = lang_utils.get_lang_lengths(lang_tensor)

            # Add start and end of sentence tokens to language
            lang_tensor, lang_lengths = lang_utils.add_sos_eos(
                lang_tensor, lang_lengths, vocab
            )
            lang_masks = lang_utils.get_lang_masks(
                lang_lengths, max_len=lang_tensor.shape[2]
            )

            self.lang[label_name] = lang_tensor
            self.lang_lengths[label_name] = lang_lengths
            self.lang_masks[label_name] = lang_masks

            # Give images their numeric ids according to alphabetical order
            if self.name == "CUB":
                img_dir = os.path.join(lang_dir, "text_c10", label_name, "*.txt")
                sorted_imgs = sorted(
                    [
                        os.path.splitext(os.path.basename(i))[0]
                        for i in glob.glob(img_dir)
                    ]
                )
                for i, img_fname in enumerate(sorted_imgs):
                    self.image_name_idx[img_fname] = i

        for x, y in zip(self.meta["image_names"], self.meta["image_labels"]):
            if y in self.sub_meta:
                self.sub_meta[y].append(x)
                label_name = self.meta["label_names"][y]

                image_basename = os.path.splitext(os.path.basename(x))[0]
                if self.name == "CUB":
                    image_lang_idx = self.image_name_idx[image_basename]
                else:
                    image_lang_idx = int(image_basename[-1])

                captions = self.lang[label_name][image_lang_idx]
                lengths = self.lang_lengths[label_name][image_lang_idx]
                masks = self.lang_masks[label_name][image_lang_idx]

                self.sub_meta_lang[y].append(captions)
                self.sub_meta_lang_length[y].append(lengths)
                self.sub_meta_lang_mask[y].append(masks)
            else:
                assert self.max_class is not None

        if self.args.scramble_lang:
            # For each class, shuffle captions for each image
            (
                self.sub_meta_lang,
                self.sub_meta_lang_length,
                self.sub_meta_lang_mask,
            ) = lang_utils.shuffle_lang_class(
                self.sub_meta_lang, self.sub_meta_lang_length, self.sub_meta_lang_mask
            )

        if self.args.scramble_lang_class:
            raise NotImplementedError

        if self.args.scramble_all:
            # Shuffle captions completely randomly
            (
                self.sub_meta_lang,
                self.sub_meta_lang_length,
                self.sub_meta_lang_mask,
            ) = lang_utils.shuffle_all_class(
                self.sub_meta_lang, self.sub_meta_lang_length, self.sub_meta_lang_mask
            )

        if self.max_img_per_class is not None:
            # Trim number of images available per class
            for cl in self.sub_meta.keys():
                self.sub_meta[cl] = self.sub_meta[cl][: self.max_img_per_class]
                self.sub_meta_lang[cl] = self.sub_meta_lang[cl][
                    : self.max_img_per_class
                ]
                self.sub_meta_lang_length[cl] = self.sub_meta_lang_length[cl][
                    : self.max_img_per_class
                ]
                self.sub_meta_lang_mask[cl] = self.sub_meta_lang_mask[cl][
                    : self.max_img_per_class
                ]

        if self.max_lang_per_class is not None:
            # Trim language available for each class; recycle language if not enough
            for cl in self.sub_meta.keys():
                self.sub_meta_lang[cl] = lang_utils.recycle_lang(
                    self.sub_meta_lang[cl], self.max_lang_per_class
                )
                self.sub_meta_lang_length[cl] = lang_utils.recycle_lang(
                    self.sub_meta_lang_length[cl], self.max_lang_per_class
                )
                self.sub_meta_lang_mask[cl] = lang_utils.recycle_lang(
                    self.sub_meta_lang_mask[cl], self.max_lang_per_class
                )

        self.sub_dataloader = []
        sub_data_loader_params = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # use main thread only or may receive multiple batches
            pin_memory=False,
        )
        for i, cl in enumerate(self.cl_list):
            sub_dataset = SubDataset(
                self.name,
                self.sub_meta[cl],
                cl,
                sub_meta_lang=self.sub_meta_lang[cl],
                sub_meta_lang_length=self.sub_meta_lang_length[cl],
                sub_meta_lang_mask=self.sub_meta_lang_mask[cl],
                transform=transform,
                n_caption=self.args.n_caption,
                args=self.args,
                max_lang_per_class=self.max_lang_per_class,
            )
            self.sub_dataloader.append(
                torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params)
            )

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)


class SubDataset:
    def __init__(
        self,
        name,
        sub_meta,
        cl,
        sub_meta_lang=None,
        sub_meta_lang_length=None,
        sub_meta_lang_mask=None,
        transform=transforms.ToTensor(),
        target_transform=identity,
        n_caption=10,
        args=None,
        max_lang_per_class=None,
    ):
        self.name = name
        self.sub_meta = sub_meta
        self.sub_meta_lang = sub_meta_lang
        self.sub_meta_lang_length = sub_meta_lang_length
        self.sub_meta_lang_mask = sub_meta_lang_mask
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform
        if not (1 <= n_caption <= 10):
            raise ValueError("Invalid # captions {}".format(n_caption))
        self.n_caption = n_caption
        cl_path = os.path.split(self.sub_meta[0])[0]
        self.img = dict(np.load(os.path.join(cl_path, "img.npz")))

        # Used if sampling from class
        self.args = args
        self.max_lang_per_class = max_lang_per_class

    def __getitem__(self, i):
        image_path = self.sub_meta[i]
        img = self.img[image_path]
        img = self.transform(img)
        target = self.target_transform(self.cl)

        if self.n_caption == 1:
            lang_idx = 0
        else:
            lang_idx = random.randint(min(self.n_caption, len(self.sub_meta_lang[i])))

        if self.args.sample_class_lang:
            # Sample from all language, rather than the ith image
            if self.max_lang_per_class is None:
                max_i = len(self.sub_meta_lang)
            else:
                max_i = min(self.max_lang_per_class, len(self.sub_meta_lang))
            which_img_lang_i = random.randint(0, max_i)
        else:
            which_img_lang_i = i

        lang = self.sub_meta_lang[which_img_lang_i][lang_idx]
        lang_length = self.sub_meta_lang_length[which_img_lang_i][lang_idx]
        lang_mask = self.sub_meta_lang_mask[which_img_lang_i][lang_idx]

        return img, target, (lang, lang_length, lang_mask)

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[: self.n_way]
