"""
Training script for solely image-captioning based tasks
"""

import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from utils import (
    AverageMeter,
    save_defaultdict_to_fs,
)
import utils
from datasets import ShapeWorld
from datasets import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
import datasets
from models import ImageRep, TextRep, TextProposal, ExWrapper
from models import MultimodalRep
from models import DotPScorer, BilinearScorer
from vision import Conv4NP, Conv4NP2, ResNet18
from tre import AddComp, MulComp, CosDist, L1Dist, L2Dist, tre
import bleu

TRE_COMP_FNS = {
    'add': AddComp,
    'mul': MulComp,
}

TRE_ERR_FNS = {
    'cos': CosDist,
    'l1': L1Dist,
    'l2': L2Dist,
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, help='Output directory')
    parser.add_argument('--backbone',
                        choices=['vgg16_fixed', 'conv4', 'conv4_2', 'resnet18'],
                        default='vgg16_fixed',
                        help='Image model')
    parser.add_argument('--max_train',
                        type=int,
                        default=None,
                        help='Max number of training examples')
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='Train batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Train epochs')
    parser.add_argument(
        '--data_dir',
        default=None,
        help='Specify custom data directory (must have shapeworld folder)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='Learning rate')
    parser.add_argument('--optimizer',
                        choices=['adam', 'rmsprop', 'sgd'],
                        default='adam',
                        help='Optimizer to use')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--language_filter',
                        default=None,
                        type=str,
                        choices=['color', 'nocolor'],
                        help='Filter language')
    parser.add_argument('--shuffle_words',
                        action='store_true',
                        help='Shuffle words for each caption')
    parser.add_argument('--shuffle_captions',
                        action='store_true',
                        help='Shuffle captions for each class')
    parser.add_argument('--bleu_interval',
                        type=int,
                        default=5,
                        help='How often to evaluate BLEU (it can be slower)')
    parser.add_argument('--log_interval',
                        type=int,
                        default=10,
                        help='How often to log loss')
    parser.add_argument('--cuda',
                        action='store_true',
                        help='Enables CUDA training')
    args = parser.parse_args()

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    # train dataset will return (image, label, hint_input, hint_target, hint_length)
    precomputed_features = args.backbone == 'vgg16_fixed'
    preprocess = args.backbone == 'resnet18'
    train_dataset = ShapeWorld(
        split='train',
        vocab=None,
        augment=True,
        precomputed_features=precomputed_features,
        max_size=args.max_train,
        preprocess=preprocess,
        data_dir=args.data_dir,
        language_filter=args.language_filter,
        shuffle_words=args.shuffle_words,
        shuffle_captions=args.shuffle_captions)
    train_vocab = train_dataset.vocab
    train_vocab_size = train_dataset.vocab_size
    train_max_length = train_dataset.max_length
    train_w2i, train_i2w = train_vocab['w2i'], train_vocab['i2w']
    pad_index = train_w2i[PAD_TOKEN]
    sos_index = train_w2i[SOS_TOKEN]
    eos_index = train_w2i[EOS_TOKEN]
    val_dataset = ShapeWorld(split='val',
                             precomputed_features=precomputed_features,
                             vocab=train_vocab,
                             preprocess=preprocess,
                             data_dir=args.data_dir)
    test_dataset = ShapeWorld(split='test',
                              precomputed_features=precomputed_features,
                              vocab=train_vocab,
                              preprocess=preprocess,
                              data_dir=args.data_dir)
    try:
        val_same_dataset = ShapeWorld(
            split='val_same',
            precomputed_features=precomputed_features,
            vocab=train_vocab,
            preprocess=preprocess,
            data_dir=args.data_dir)
        test_same_dataset = ShapeWorld(
            split='test_same',
            precomputed_features=precomputed_features,
            vocab=train_vocab,
            preprocess=preprocess,
            data_dir=args.data_dir)
        has_same = True
    except RuntimeError:
        has_same = False

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False)
    if has_same:
        val_same_loader = torch.utils.data.DataLoader(
            val_same_dataset, batch_size=args.batch_size, shuffle=False)
        test_same_loader = torch.utils.data.DataLoader(
            test_same_dataset, batch_size=args.batch_size, shuffle=False)

    data_loader_dict = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'val_same': val_same_loader if has_same else None,
        'test_same': test_same_loader if has_same else None,
    }

    if args.backbone == 'vgg16_fixed':
        backbone_model = None
    elif args.backbone == 'conv4':
        backbone_model = Conv4NP()
    elif args.backbone == 'conv4_2':
        backbone_model = Conv4NP2()
    elif args.backbone == 'resnet18':
        backbone_model = ResNet18()
    else:
        raise NotImplementedError(args.backbone)

    image_model = ExWrapper(ImageRep(backbone_model))
    image_model = image_model.to(device)
    params_to_optimize = list(image_model.parameters())

    embedding_model = nn.Embedding(train_vocab_size, 512)

    proposal_model = TextProposal(embedding_model)
    proposal_model = proposal_model.to(device)
    params_to_optimize.extend(proposal_model.parameters())

    optfunc = {
        'adam': optim.Adam,
        'rmsprop': optim.RMSprop,
        'sgd': optim.SGD
    }[args.optimizer]
    optimizer = optfunc(params_to_optimize, lr=args.lr)

    def train(epoch, n_steps=100):
        image_model.train()
        proposal_model.train()

        loss_total = 0
        pbar = tqdm(total=n_steps)
        for batch_idx in range(n_steps):
            examples, image, label, hint_seq, hint_length, *rest = \
                train_dataset.sample_train(args.batch_size)

            examples = examples.to(device)
            image = image.to(device)
            label = label.to(device)
            batch_size = len(image)
            n_ex = examples.shape[1]

            # Load hint
            hint_seq = hint_seq.to(device)
            hint_length = hint_length.to(device)
            max_hint_length = hint_length.max().item()
            # Cap max length if it doesn't fill out the tensor
            if max_hint_length != hint_seq.shape[1]:
                hint_seq = hint_seq[:, :max_hint_length]

            # Learn representations of images and examples
            examples_rep = image_model(examples)
            examples_rep_mean = torch.mean(examples_rep, dim=1)

            # Hypothesis loss
            # TODO: Decode from concepts vs decode from images
            hyp_source_rep = examples_rep_mean
            hyp_batch_size = batch_size

            # Decode images/examples to hints
            hypo_out = proposal_model(hyp_source_rep, hint_seq,
                                      hint_length)
            seq_len = hint_seq.size(1)
            hypo_out = hypo_out[:, :-1].contiguous()
            hint_seq = hint_seq[:, 1:].contiguous()

            hypo_out_2d = hypo_out.view(hyp_batch_size * (seq_len - 1),
                                        train_vocab_size)
            hint_seq_2d = hint_seq.long().view(hyp_batch_size * (seq_len - 1))
            hypo_loss = F.cross_entropy(hypo_out_2d,
                                        hint_seq_2d,
                                        reduction='none')
            # NOTE: Need to mask out!!
            hypo_loss = hypo_loss.view(hyp_batch_size, (seq_len - 1))
            hypo_loss = torch.mean(torch.sum(hypo_loss, dim=1))

            loss_total += hypo_loss.item()

            optimizer.zero_grad()
            hypo_loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                pbar.set_description('Epoch {} Loss: {:.6f}'.format(
                    epoch, hypo_loss.item()))
                pbar.refresh()

            pbar.update()
        pbar.close()
        print('====> {:>12}\tEpoch: {:>3}\tLoss: {:.4f}'.format(
            '(train)', epoch, loss_total))

        return loss_total

    def test(epoch, split='train'):
        image_model.eval()
        proposal_model.eval()

        bleu_meter = AverageMeter()
        em_meter = AverageMeter()
        data_loader = data_loader_dict[split]

        pred_caps = []
        true_caps = []

        with torch.no_grad():
            for examples, image, label, true_hint_seq, true_hint_length, *rest in data_loader:
                examples = examples.to(device)
                image = image.to(device)
                batch_size = len(image)

                examples_rep = image_model(examples)
                examples_rep_mean = torch.mean(examples_rep, dim=1)

                hint_seq, hint_length = proposal_model.sample(
                    examples_rep_mean,
                    sos_index,
                    eos_index,
                    pad_index,
                    greedy=True)

                hint_lang = train_dataset.to_text(hint_seq)
                true_hint_lang = train_dataset.to_text(true_hint_seq)
                true_hint_langs = datasets.compute_alternatives(true_hint_lang)

                # Cut off sos, eos tokens
                hint_lang = list(map(datasets.remove_special_tokens, hint_lang))
                true_hint_langs = [list(map(datasets.remove_special_tokens, hls)) for hls in true_hint_langs]

                em = np.array([h in th for h, th in zip(hint_lang, true_hint_langs)]).mean()
                b = bleu.compute_bleu(true_hint_langs, hint_lang)[0]
                # Prefer *100
                b *= 100

                bleu_meter.update(b, batch_size)
                em_meter.update(em, batch_size)

                pred_caps.extend(hint_lang)
                true_caps.extend(true_hint_langs)

        print('====> {:>12}\tEpoch: {:>3}\tBLEU: {:.4f}\tEM: {:.4f}'.format(
            '({})'.format(split), epoch, bleu_meter.avg, em_meter.avg))

        return bleu_meter.avg, em_meter.avg, pred_caps, true_caps

    best_epoch = 0
    best_epoch_bleu = 0
    best_val_bleu = 0
    best_val_same_bleu = 0
    best_test_bleu = 0
    best_test_same_bleu = 0
    best_val_em = 0
    best_val_same_em = 0
    best_test_em = 0
    best_test_same_em = 0
    metrics = defaultdict(lambda: [])

    save_defaultdict_to_fs(vars(args), os.path.join(args.exp_dir, 'args.json'))
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)

        if epoch % args.bleu_interval == 0:
            train_bleu, train_em, *_ = test(epoch, 'train')
            val_bleu, val_em, val_pred, val_trues = test(epoch, 'val')

            test_bleu, test_em, *_ = test(epoch, 'test')
            if has_same:
                val_same_bleu, val_same_em, val_same_pred, val_same_trues = test(epoch, 'val_same')
                test_same_bleu, test_same_em, *_ = test(epoch, 'test_same')
                val_pred += val_same_pred
                val_trues += val_same_trues
            else:
                val_same_bleu = val_bleu
                test_same_bleu = test_bleu
                val_same_em = val_em
                test_same_em = test_em

            # Compute confidence intervals
            # TODO
            n_test = len(val_pred)

            epoch_bleu = (val_bleu + val_same_bleu) / 2
            is_best_epoch = epoch_bleu > best_val_bleu
            if is_best_epoch:
                best_epoch = epoch
                best_epoch_bleu = epoch_bleu

                best_val_bleu = val_bleu
                best_val_same_bleu = val_same_bleu
                best_test_bleu = test_bleu
                best_test_same_bleu = test_same_bleu

                best_val_em = val_em
                best_val_same_em = val_same_em
                best_test_em = test_em
                best_test_same_em = test_same_em
                # Save best predictions
                utils.save_predictions(val_pred, val_trues, args.exp_dir,
                                       filename='val_predictions.csv')

            metrics['epoch'].append(epoch)
            metrics['train_bleu'].append(train_bleu)
            metrics['val_bleu'].append(val_bleu)
            metrics['val_same_bleu'].append(val_same_bleu)
            metrics['test_bleu'].append(test_bleu)
            metrics['test_same_bleu'].append(test_same_bleu)
            metrics['train_em'].append(train_em)
            metrics['val_em'].append(val_em)
            metrics['val_same_em'].append(val_same_em)
            metrics['test_em'].append(test_em)
            metrics['test_same_em'].append(test_same_em)

            metrics = dict(metrics)
            # Assign best accs
            metrics['best_epoch'] = best_epoch
            metrics['best_val_bleu'] = best_val_bleu
            metrics['best_val_same_bleu'] = best_val_same_bleu
            metrics['best_test_bleu'] = best_test_bleu
            metrics['best_test_same_bleu'] = best_test_same_bleu
            metrics['best_val_em'] = best_val_em
            metrics['best_val_same_em'] = best_val_same_em
            metrics['best_test_em'] = best_test_em
            metrics['best_test_same_em'] = best_test_same_em
            metrics['has_same'] = has_same
            save_defaultdict_to_fs(metrics,
                                   os.path.join(args.exp_dir, 'metrics.json'))

    print('====> DONE')
    print('====> BEST EPOCH: {}'.format(best_epoch))
    print('====> {:>17}\tEpoch: {}\tBLEU: {:.4f}\tEM: {:.4f}'.format(
        '(best_val)', best_epoch, best_val_bleu, best_val_em))
    print('====> {:>17}\tEpoch: {}\tBLEU: {:.4f}\tEM: {:.4f}'.format(
        '(best_val_same)', best_epoch, best_val_same_bleu, best_val_same_em))
    print('====> {:>17}\tEpoch: {}\tBLEU: {:.4f}\tEM {:.4f}'.format(
        '(best_test)', best_epoch, best_test_bleu, best_test_em))
    print('====> {:>17}\tEpoch: {}\tBLEU: {:.4f}\tEM {:.4f}'.format(
        '(best_test_same)', best_epoch, best_test_same_bleu, best_test_same_em))
    print('====>')
    print('====> {:>17}\tEpoch: {}\tBLEU: {:.4f}\tEM: {:.4f}'.format(
        '(best_val_avg)', best_epoch, (best_val_bleu + best_val_same_bleu) / 2,
    (best_val_em + best_val_same_em) / 2))
    print('====> {:>17}\tEpoch: {}\tBLEU: {:.4f}\tEM: {:.4f}'.format(
        '(best_test_avg)', best_epoch,
        (best_test_bleu + best_test_same_bleu) / 2,
    (best_test_em + best_test_same_em) / 2))
