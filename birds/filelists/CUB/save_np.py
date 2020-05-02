"""
For each class, load images and save as numpy arrays.
"""

import os

import numpy as np
from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Save numpy", formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--cub_dir", default="CUB_200_2011/images", help="Directory to load/cache"
    )
    parser.add_argument(
        "--original_cub_dir",
        default="CUB_200_2011/images",
        help="Original CUB directory if you want the image keys to be different (in case --cub_dir has changed)",
    )
    parser.add_argument("--filelist_prefix", default="./filelists/CUB/")

    args = parser.parse_args()

    for bird_class in tqdm(os.listdir(args.cub_dir), desc="Classes"):
        bird_imgs_np = {}
        class_dir = os.path.join(args.cub_dir, bird_class)
        bird_imgs = sorted([x for x in os.listdir(class_dir) if x != "img.npz"])
        for bird_img in bird_imgs:
            bird_img_fname = os.path.join(class_dir, bird_img)
            img = Image.open(bird_img_fname).convert("RGB")
            img_np = np.asarray(img)

            full_bird_img_fname = os.path.join(
                args.filelist_prefix, args.original_cub_dir, bird_class, bird_img
            )

            bird_imgs_np[full_bird_img_fname] = img_np

        np_fname = os.path.join(class_dir, "img.npz")
        np.savez_compressed(np_fname, **bird_imgs_np)
