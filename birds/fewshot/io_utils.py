"""
Contains argument parsers and utilities for saving and loading metrics and
models.
"""

import argparse
import glob
import os

import numpy as np

import backbone


model_dict = dict(
    Conv4=backbone.Conv4,
    Conv4NP=backbone.Conv4NP,
    Conv4S=backbone.Conv4S,
    Conv6=backbone.Conv6,
    ResNet10=backbone.ResNet10,
    ResNet18=backbone.ResNet18,
    PretrainedResNet18=backbone.PretrainedResNet18,
    ResNet34=backbone.ResNet34,
    ResNet50=backbone.ResNet50,
    ResNet101=backbone.ResNet101,
)


def parse_args(script):
    parser = argparse.ArgumentParser(
        description="few-shot script %s" % (script),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Specify checkpoint dir (if none, automatically generate)",
    )
    parser.add_argument("--model", default="Conv4", help="Choice of backbone")
    parser.add_argument("--lsl", action="store_true")
    parser.add_argument(
        "--l3", action="store_true", help="Use l3 (do not need to --lsl)"
    )
    parser.add_argument("--l3_n_infer", type=int, default=10, help="Number to sample")
    parser.add_argument(
        "--rnn_type", choices=["gru", "lstm"], default="gru", help="Language RNN type"
    )
    parser.add_argument(
        "--rnn_num_layers", default=1, type=int, help="Language RNN num layers"
    )
    parser.add_argument(
        "--rnn_dropout", default=0.0, type=float, help="Language RNN dropout"
    )
    parser.add_argument(
        "--lang_supervision",
        default="class",
        choices=["instance", "class"],
        help="At what level to supervise with language?",
    )
    parser.add_argument("--glove_init", action="store_true")
    parser.add_argument(
        "--freeze_emb", action="store_true", help="Freeze LM word embedding layer"
    )

    langparser = parser.add_argument_group("language settings")
    langparser.add_argument(
        "--shuffle_lang", action="store_true", help="Shuffle words in caption"
    )
    langparser.add_argument(
        "--scramble_lang",
        action="store_true",
        help="Scramble captions -> images mapping in a class",
    )
    langparser.add_argument(
        "--sample_class_lang",
        action="store_true",
        help="Sample language randomly from class, rather than getting lang assoc. w/ img",
    )
    langparser.add_argument(
        "--scramble_all",
        action="store_true",
        help="Scramble captions -> images mapping across all classes",
    )
    langparser.add_argument(
        "--scramble_lang_class",
        action="store_true",
        help="Scramble captions -> images mapping across all classes, but keep classes consistent",
    )
    langparser.add_argument(
        "--language_filter",
        default="all",
        choices=["all", "color", "nocolor"],
        help="What language to use",
    )

    parser.add_argument(
        "--lang_hidden_size", type=int, default=200, help="Language decoder hidden size"
    )
    parser.add_argument(
        "--lang_emb_size", type=int, default=300, help="Language embedding hidden size"
    )
    parser.add_argument(
        "--lang_lambda", type=float, default=5, help="Weight on language loss"
    )

    parser.add_argument(
        "--n_caption",
        type=int,
        default=1,
        choices=list(range(1, 11)),
        help="How many captions to use for pretraining",
    )
    parser.add_argument(
        "--max_class", type=int, default=None, help="Max number of training classes"
    )
    parser.add_argument(
        "--max_img_per_class",
        type=int,
        default=None,
        help="Max number of images per training class",
    )
    parser.add_argument(
        "--max_lang_per_class",
        type=int,
        default=None,
        help="Max number of language per training class (recycled among images)",
    )
    parser.add_argument(
        "--train_n_way", default=5, type=int, help="class num to classify for training"
    )
    parser.add_argument(
        "--test_n_way",
        default=5,
        type=int,
        help="class num to classify for testing (validation) ",
    )
    parser.add_argument(
        "--n_shot",
        default=1,
        type=int,
        help="number of labeled data in each class, same as n_support",
    )
    parser.add_argument(
        "--n_workers",
        default=4,
        type=int,
        help="Use this many workers for loading data",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Inspect generated language"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed (torch only; not numpy)"
    )

    if script == "train":
        parser.add_argument(
            "--n", default=1, type=int, help="Train run number (used for metrics)"
        )
        parser.add_argument(
            "--optimizer",
            default="adam",
            choices=["adam", "amsgrad", "rmsprop"],
            help="Optimizer",
        )
        parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
        parser.add_argument(
            "--rnn_lr_scale",
            default=1.0,
            type=float,
            help="Scale the RNN lr by this amount of the original lr",
        )
        parser.add_argument("--save_freq", default=50, type=int, help="Save frequency")
        parser.add_argument("--start_epoch", default=0, type=int, help="Starting epoch")
        parser.add_argument(
            "--stop_epoch", default=600, type=int, help="Stopping epoch"
        )  # for meta-learning methods, each epoch contains 100 episodes
        parser.add_argument(
            "--resume",
            action="store_true",
            help="continue from previous trained model with largest epoch",
        )
    elif script == "test":
        parser.add_argument(
            "--split",
            default="novel",
            choices=["base", "val", "novel"],
            help="which split to evaluate on",
        )
        parser.add_argument(
            "--save_iter",
            default=-1,
            type=int,
            help="saved feature from the model trained in x epoch, use the best model if x is -1",
        )
        parser.add_argument(
            "--save_embeddings",
            action="store_true",
            help="Save embeddings from language model, then exit (requires --lsl)",
        )
        parser.add_argument(
            "--embeddings_file",
            default="./embeddings.txt",
            help="File to save embeddings to",
        )
        parser.add_argument(
            "--embeddings_metadata",
            default="./embeddings_metadata.txt",
            help="File to save embedding metadata to (currently just words)",
        )
        parser.add_argument(
            "--record_file",
            default="./record/results.txt",
            help="Where to write results to",
        )
    else:
        raise ValueError("Unknown script")

    args = parser.parse_args()

    if "save_embeddings" in args and (args.save_embeddings and not args.lsl):
        parser.error("Must set --lsl to save embeddings")

    if args.glove_init and not (args.lsl or args.l3):
        parser.error("Must set --lsl to init with glove")

    return args


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, "{:d}.tar".format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, "*.tar"))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != "best_model.tar"]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, "{:d}.tar".format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, "best_model.tar")
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
