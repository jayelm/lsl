import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import numpy as np
import pandas as pd


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--savedir', type=str, default='../../custom_filelists/CUB/',
                        help='Directory to save filelists')

    args = parser.parse_args()

    random = np.random.RandomState(args.seed)

    filelist_path = './filelists/CUB/'
    data_path = 'CUB_200_2011/images'
    dataset_list = ['base', 'val', 'novel']

    folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    folder_list.sort()
    label_dict = dict(zip(folder_list, range(0, len(folder_list))))

    classfile_list_all = []

    # Load attributes
    attrs = pd.read_csv('./CUB_200_2011/attributes/image_attribute_labels.txt',
                        sep=' ',
                        header=None,
                        names=['image_id', 'attribute_id', 'is_present', 'certainty_id', 'time'])
    # Zero out attributes with certainty < 3
    attrs['is_present'] = np.where(attrs['certainty_id'] < 3, 0, attrs['is_present'])
    # Get image names
    image_names = pd.read_csv('./CUB_200_2011/images.txt', sep=' ',
                              header=None,
                              names=['image_id', 'image_name'])
    attrs = attrs.merge(image_names, on='image_id')
    attrs['is_present'] = attrs['is_present'].astype(str)
    attrs = attrs.groupby('image_name')['is_present'].apply(lambda col: ''.join(col))
    attrs = dict(zip(attrs.index, attrs))
    attrs = {os.path.basename(k): v for k, v in attrs.items()}

    for i, folder in enumerate(folder_list):
        folder_path = join(data_path, folder)
        classfile_list_all.append([
            join(filelist_path, folder_path, cf) for cf in listdir(folder_path)
            if (isfile(join(folder_path, cf)) and cf[0] != '.' and not cf.endswith('.npz'))
        ])
        random.shuffle(classfile_list_all[i])

    for dataset in dataset_list:
        file_list = []
        label_list = []
        for i, classfile_list in enumerate(classfile_list_all):
            if 'base' in dataset:
                if (i % 2 == 0):
                    file_list.extend(classfile_list)
                    label_list.extend(np.repeat(
                        i, len(classfile_list)).tolist())
            if 'val' in dataset:
                if (i % 4 == 1):
                    file_list.extend(classfile_list)
                    label_list.extend(np.repeat(
                        i, len(classfile_list)).tolist())
            if 'novel' in dataset:
                if (i % 4 == 3):
                    file_list.extend(classfile_list)
                    label_list.extend(np.repeat(
                        i, len(classfile_list)).tolist())

        # Get attributes
        attribute_list = [
            attrs[os.path.basename(f)] for f in file_list if not f.endswith('.npz')
        ]

        djson = {
            'label_names': folder_list,
            'image_names': file_list,
            'image_labels': label_list,
            'image_attributes': attribute_list,
        }

        os.makedirs(args.savedir, exist_ok=True)
        with open(os.path.join(args.savedir, dataset + '.json'), 'w') as fout:
            json.dump(djson, fout)

        print("%s -OK" % dataset)
