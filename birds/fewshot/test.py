"""
Test script.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data.sampler

import constants
from data import lang_utils
from data.datamgr import SetDataManager, TransformLoader
from io_utils import get_assigned_file, get_best_file, model_dict, parse_args
from models.language import TextProposal, TextRep
from models.protonet import ProtoNet

torch.backends.cudnn.enabled = False

if __name__ == "__main__":
    args = parse_args("test")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    acc_all = []

    vocab = lang_utils.load_vocab(constants.LANG_DIR)

    l3_model = None
    lang_model = None
    if args.lsl or args.l3:
        embedding_model = nn.Embedding(len(vocab), args.lang_emb_size)
        lang_model = TextProposal(
            embedding_model,
            input_size=1600,
            hidden_size=args.lang_hidden_size,
            project_input=1600 != args.lang_hidden_size,
            rnn=args.rnn_type,
            num_layers=args.rnn_num_layers,
            dropout=args.rnn_dropout,
            vocab=vocab,
            **lang_utils.get_special_indices(vocab),
        )

        if args.l3:
            l3_model = TextRep(
                embedding_model,
                hidden_size=args.lang_hidden_size,
                rnn=args.rnn_type,
                num_layers=args.rnn_num_layers,
                dropout=args.rnn_dropout,
            )
            l3_model = l3_model.cuda()

        embedding_model = embedding_model.cuda()
        lang_model = lang_model.cuda()

    model = ProtoNet(
        model_dict[args.model],
        n_way=args.test_n_way,
        n_support=args.n_shot,
        # Language options
        lsl=args.lsl,
        language_model=lang_model,
        lang_supervision=args.lang_supervision,
        l3=args.l3,
        l3_model=l3_model,
        l3_n_infer=args.l3_n_infer,
    )

    model = model.cuda()

    if args.save_iter != -1:
        modelfile = get_assigned_file(args.checkpoint_dir, args.save_iter)
    else:
        modelfile = get_best_file(args.checkpoint_dir)

    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(
            tmp["state"],
            # If language was used for pretraining, ignore
            # the language model component here. If we want to use language,
            # make sure the model is loaded
            strict=args.lsl,
        )

        if args.save_embeddings:
            if args.lsl:
                weights = model.language_model.embedding.weight.detach().cpu().numpy()
                vocab_srt = sorted(list(vocab.items()), key=lambda x: x[1])
                vocab_srt = [v[0] for v in vocab_srt]
                with open(args.embeddings_file, "w") as fout:
                    fout.write("\n".join(vocab_srt))
                    fout.write("\n")
                np.savetxt(args.embeddings_metadata, weights, fmt="%f", delimiter="\t")
                sys.exit(0)

    # Run the test loop for 600 iterations
    ITER_NUM = 600
    N_QUERY = 15

    test_datamgr = SetDataManager(
        "CUB",
        84,
        n_query=N_QUERY,
        n_way=args.test_n_way,
        n_support=args.n_shot,
        n_episode=ITER_NUM,
        args=args,
    )
    test_loader = test_datamgr.get_data_loader(
        os.path.join(constants.DATA_DIR, f"{args.split}.json"),
        aug=False,
        lang_dir=constants.LANG_DIR,
        normalize=False,
        vocab=vocab,
    )
    normalizer = TransformLoader(84).get_normalize()

    model.eval()

    acc_all = model.test_loop(
        test_loader,
        normalizer=normalizer,
        verbose=True,
        return_all=True,
        # Debug on first loop only
        debug=args.debug,
        debug_dir=os.path.split(args.checkpoint_dir)[0],
    )
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print(
        "%d Test Acc = %4.2f%% +- %4.2f%%"
        % (ITER_NUM, acc_mean, 1.96 * acc_std / np.sqrt(ITER_NUM))
    )

    with open(args.record_file, "a") as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        acc_ci = 1.96 * acc_std / np.sqrt(ITER_NUM)
        f.write(
            json.dumps(
                {
                    "time": timestamp,
                    "split": args.split,
                    "setting": args.checkpoint_dir,
                    "iter_num": ITER_NUM,
                    "acc": acc_mean,
                    "acc_ci": acc_ci,
                    "acc_all": list(acc_all),
                    "acc_std": acc_std,
                },
                sort_keys=True,
            )
        )
        f.write("\n")
