# This code is modified from
# https://github.com/facebookresearch/low-shot-shrink-hallucinate

from abc import abstractmethod

import torch
import torchvision.transforms as transforms

import data.additional_transforms as add_transforms
from data.dataset import EpisodicBatchSampler, SetDataset, SimpleDataset


class TransformLoader:
    def __init__(
        self,
        image_size,
        normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4),
    ):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == "ImageJitter":
            method = add_transforms.ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type == "RandomResizedCrop":
            return method(self.image_size)
        elif transform_type == "CenterCrop":
            return method(self.image_size)
        elif transform_type == "Resize":
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == "Normalize":
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(
        self,
        aug=False,
        normalize=True,
        to_pil=True,
        confound_noise=0.0,
        confound_noise_class_weight=0.0,
    ):
        if aug:
            transform_list = [
                "RandomResizedCrop",
                "ImageJitter",
                "RandomHorizontalFlip",
                "ToTensor",
            ]
        else:
            transform_list = ["Resize", "CenterCrop", "ToTensor"]

        if confound_noise != 0.0:
            transform_list.append(
                ("Noise", confound_noise, confound_noise_class_weight)
            )

        if normalize:
            transform_list.append("Normalize")

        if to_pil:
            transform_list = ["ToPILImage"] + transform_list

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

    def get_normalize(self):
        return self.parse_transform("Normalize")


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size, num_workers=12):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        self.num_workers = num_workers

    def get_data_loader(
        self, data_file, aug, lang_dir=None, normalize=True, to_pil=False
    ):  # parameters that would change on train/val set
        if lang_dir is not None:
            raise NotImplementedError
        transform = self.trans_loader.get_composed_transform(
            aug, normalize=normalize, to_pil=to_pil
        )
        dataset = SimpleDataset(data_file, transform)
        data_loader_params = dict(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader


class SetDataManager(DataManager):
    def __init__(
        self, name, image_size, n_way, n_support, n_query, n_episode=100, args=None
    ):
        super(SetDataManager, self).__init__()
        self.name = name
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode
        self.args = args

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(
        self,
        data_file,
        aug,
        lang_dir=None,
        normalize=True,
        vocab=None,
        max_class=None,
        max_img_per_class=None,
        max_lang_per_class=None,
    ):
        transform = self.trans_loader.get_composed_transform(aug, normalize=normalize)

        dataset = SetDataset(
            self.name,
            data_file,
            self.batch_size,
            transform,
            args=self.args,
            lang_dir=lang_dir,
            vocab=vocab,
            max_class=max_class,
            max_img_per_class=max_img_per_class,
            max_lang_per_class=max_lang_per_class,
        )
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        data_loader_params = dict(
            batch_sampler=sampler, num_workers=self.args.n_workers, pin_memory=True,
        )
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader
