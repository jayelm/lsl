"""
Training script
"""

import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, recall_score, precision_score
from torchtext.data.metrics import bleu_score
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from utils import (
    AverageMeter,
    save_defaultdict_to_fs,
    idx2word,
)

from arguments import ArgumentParser
from bertadam import BertAdam
from datasets import ShapeWorld, extract_features
from datasets import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from lxmert import Lxmert
from models import ImageRep, TextRep, TextProposal, ExWrapper
from models import MultimodalRep,MultimodalDeepRep
from models import DotPScorer, BilinearScorer
from vision import Conv4NP, ResNet18
from tre import AddComp, MulComp, CosDist, L1Dist, L2Dist, tre
from retrievers import construct_dict, gen_retriever
import matplotlib.pyplot as plt

TRE_COMP_FNS = {
    'add': AddComp,
    'mul': MulComp,
}

TRE_ERR_FNS = {
    'cos': CosDist,
    'l1': L1Dist,
    'l2': L2Dist,
}


def combine_feats(all_feats):
    """
    Combine feats like language, mask them, and get vocab
    """
    vocab = {}
    max_feat_len = max(len(f) for f in all_feats)
    feats_t = torch.zeros(len(all_feats), max_feat_len, dtype=torch.int64)
    feats_mask = torch.zeros(len(all_feats), max_feat_len, dtype=torch.uint8)
    for feat_i, feat in enumerate(all_feats):
        for j, f in enumerate(feat):
            if f not in vocab:
                vocab[f] = len(vocab)
            i_f = vocab[f]
            feats_t[feat_i, j] = i_f
            feats_mask[feat_i, j] = 1
    return feats_t, feats_mask, vocab


if __name__ == "__main__":
    args = ArgumentParser().parse_args()

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    # train dataset will return (image, label, hint_input, hint_target, hint_length)
    precomputed_features = args.backbone == 'vgg16_fixed'
    preprocess = args.backbone == 'resnet18' or args.backbone == 'lxmert'
    train_dataset = ShapeWorld(
        split='train',
        vocab=None,
        augment=True,
        precomputed_features=precomputed_features,
        max_size=args.max_train,
        preprocess=preprocess,
        noise=args.noise,
        class_noise_weight=args.class_noise_weight,
        fixed_noise_colors=args.fixed_noise_colors,
        fixed_noise_colors_max_rgb=args.fixed_noise_colors_max_rgb,
        noise_type=args.noise_type,
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
    test_class_noise_weight = 0.0
    if args.noise_at_test:
        test_noise = args.noise
    else:
        test_noise = 0.0
    val_dataset = ShapeWorld(split='val',
                             precomputed_features=precomputed_features,
                             vocab=train_vocab,
                             preprocess=preprocess,
                             noise=test_noise,
                             class_noise_weight=0.0,
                             noise_type=args.noise_type,
                             data_dir=args.data_dir)
    test_dataset = ShapeWorld(split='test',
                              precomputed_features=precomputed_features,
                              vocab=train_vocab,
                              preprocess=preprocess,
                              noise=test_noise,
                              class_noise_weight=0.0,
                              noise_type=args.noise_type,
                              data_dir=args.data_dir)
    try:
        val_same_dataset = ShapeWorld(
            split='val_same',
            precomputed_features=precomputed_features,
            vocab=train_vocab,
            preprocess=preprocess,
            noise=test_noise,
            class_noise_weight=0.0,
            noise_type=args.noise_type,
            data_dir=args.data_dir)
        test_same_dataset = ShapeWorld(
            split='test_same',
            precomputed_features=precomputed_features,
            vocab=train_vocab,
            preprocess=preprocess,
            noise=test_noise,
            class_noise_weight=0.0,
            noise_type=args.noise_type,
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
    elif args.backbone == 'resnet18':
        backbone_model = ResNet18()
    elif args.backbone == 'lxmert':
        backbone_model = Lxmert(train_vocab_size, 768, 9408, 4, args.initializer_range, pretrained=False)
    else:
        raise NotImplementedError(args.backbone)

    if args.hint_retriever:
        image_model = ExWrapper(ImageRep(backbone_model, hidden_size=512), retrieve_mode=True)
    elif args.backbone == 'lxmert':
        image_model = backbone_model
    else:
        image_model = ExWrapper(ImageRep(backbone_model, hidden_size=512))
    image_model = image_model.to(device)
    params_to_optimize = list(image_model.parameters())

    if args.comparison == 'dotp':
        scorer_model = DotPScorer()
    elif args.comparison == 'bilinear':
        # FIXME: This won't work with --poe
        scorer_model = BilinearScorer(512,
                                      dropout=args.dropout,
                                      identity_debug=args.debug_bilinear)
    else:
        raise NotImplementedError
    scorer_model = scorer_model.to(device)
    params_to_optimize.extend(scorer_model.parameters())

    if args.use_hyp:
        embedding_model = nn.Embedding(train_vocab_size, 512)

    if args.decode_hyp:
        proposal_model = TextProposal(embedding_model, hidden_size=512)
        proposal_model = proposal_model.to(device)
        params_to_optimize.extend(proposal_model.parameters())

    if args.encode_hyp:
        if args.hint_retriever:
            hint_model = TextRep(embedding_model, hidden_size=512, retrieve_mode=True)
        else:
            hint_model = TextRep(embedding_model, hidden_size=512)
        hint_model = hint_model.to(device)
        params_to_optimize.extend(hint_model.parameters())

    if args.multimodal_concept:
        multimodal_model = MultimodalDeepRep()
        multimodal_model = multimodal_model.to(device)
        params_to_optimize.extend(multimodal_model.parameters())

    optfunc = {
        'adam': optim.Adam,
        'rmsprop': optim.RMSprop,
        'sgd': optim.SGD,
        'bertadam': BertAdam
    }[args.optimizer]

    t_total = int(100 * args.epochs)
    optimizer = optfunc(params_to_optimize, lr=args.lr, warmup=args.warmup_ratio, t_total=t_total)

    # initialize weight and bias
    wandb.init(project='v_dev', entity='lsl')
    config = wandb.config
    config.learning_rate = args.lr

    wandb.watch(image_model)

    def train(epoch, n_steps=100):
        image_model.train()
        scorer_model.train()
        if args.decode_hyp:
            proposal_model.train()
        if args.encode_hyp:
            hint_model.train()
        if args.multimodal_concept:
            multimodal_model.train()

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

            if args.use_hyp:
                # Load hint
                hint_seq = hint_seq.to(device)
                hint_length = hint_length.to(device)
                max_hint_length = hint_length.max().item()
                # Cap max length if it doesn't fill out the tensor
                if max_hint_length != hint_seq.shape[1]:
                    hint_seq = hint_seq[:, :max_hint_length]

            # Learn representations of images and examples
            image_rep = image_model(image)
            examples_rep = image_model(examples)
            examples_rep_mean = torch.mean(examples_rep, dim=1)
            
            # Prediction loss
            if args.infer_hyp:
                # Use hypothesis to compute prediction loss
                # (how well does true hint match image repr)?
                if args.scheduled_sampling:
                    use_truth_prob = 1 - ((batch_idx + 1) + n_steps * (epoch - 1)) / n_steps / args.epochs

                    use_truth = np.random.choice(a=[True, False], p=[use_truth_prob, 1 - use_truth_prob])
                    if not use_truth:
                        hint_modified_seq, hint_modified_length = proposal_model.sample(
                            examples_rep_mean,
                            sos_index,
                            eos_index,
                            pad_index,
                            greedy=False)
                        hint_modified_seq = hint_modified_seq.to(device)
                        hint_modified_length = hint_modified_length.to(device)
                        hint_rep = hint_model(hint_modified_seq, hint_modified_length)
                    else:
                        hint_rep = hint_model(hint_seq, hint_length)

                else:
                    hint_rep = hint_model(hint_seq, hint_length)

                if args.multimodal_concept:
                    hint_rep = multimodal_model(hint_rep, examples_rep_mean)

                score = scorer_model.score(hint_rep, image_rep)

                if args.poe:
                    image_score = scorer_model.score(examples_rep_mean,
                                                     image_rep)
                    score = score + image_score
                pred_loss = F.binary_cross_entropy_with_logits(
                    score, label.float())
            else:
                # Use concept to compute prediction loss
                # (how well does example repr match image repr)?
                score = scorer_model.score(examples_rep_mean, image_rep)
                pred_loss = F.binary_cross_entropy_with_logits(
                    score, label.float())

            # Hypothesis loss
            if args.use_hyp:
                # How plausible is the true hint under example/image rep?
                if args.predict_image_hyp:
                    # Use raw images, flatten out tasks
                    hyp_batch_size = batch_size * n_ex
                    hyp_source_rep = examples_rep.view(hyp_batch_size, -1)
                    hint_seq = hint_seq.unsqueeze(1).repeat(1, n_ex, 1).view(
                        hyp_batch_size, -1)
                    hint_length = hint_length.unsqueeze(1).repeat(
                        1, n_ex).view(hyp_batch_size)
                else:
                    hyp_source_rep = examples_rep_mean
                    hyp_batch_size = batch_size

                if args.predict_hyp and args.predict_hyp_task == 'embed':
                    # Encode hints, minimize distance between hint and images/examples
                    hint_rep = hint_model(hint_seq, hint_length)
                    dists = torch.norm(hyp_source_rep - hint_rep, p=2, dim=1)
                    hypo_loss = torch.mean(dists)
                else:
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
                    hypo_loss = hypo_loss.view(hyp_batch_size, (seq_len - 1))
                    hypo_loss = torch.mean(torch.sum(hypo_loss, dim=1))

                loss = args.pred_lambda * pred_loss + args.hypo_lambda * hypo_loss
            else:
                loss = pred_loss

            loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                pbar.set_description('Epoch {} Loss: {:.6f}'.format(
                    epoch, loss.item()))
                pbar.refresh()

            pbar.update()
        pbar.close()
        print('====> {:>12}\tEpoch: {:>3}\tLoss: {:.4f}'.format(
            '(train)', epoch, loss_total))

        return loss_total

    def test(epoch, split='train', hint_rep_dict=None):
        image_model.eval()
        scorer_model.eval()
        if args.infer_hyp:
            # If predicting hyp only, ignore encode/decode models for eval
            proposal_model.eval()
            hint_model.eval()
            if args.multimodal_concept:
                multimodal_model.eval()

        accuracy_meter = AverageMeter(raw=True)
        precision_meter = AverageMeter(raw=True)
        recall_meter = AverageMeter(raw=True)
        retrival_acc_meter = AverageMeter(raw=True)
        bleu_meter_n1 = AverageMeter(raw=True)
        bleu_meter_n2 = AverageMeter(raw=True)
        bleu_meter_n3 = AverageMeter(raw=True)
        bleu_meter_n4 = AverageMeter(raw=True)
        data_loader = data_loader_dict[split]

        with torch.no_grad():
            idx = 0
            for examples, image, label, hint_seq, hint_length, *rest in data_loader:
                if idx > len(data_loader) // 2:
                    break
                idx += 1
                
                examples = examples.to(device)
                image = image.to(device)
                label = label.to(device)
                label_np = label.cpu().numpy().astype(np.uint8)
                batch_size = len(image)

                image_rep = image_model(image)

                if not args.oracle or args.multimodal_concept or args.poe:
                    # Compute example representation
                    examples_rep = image_model(examples)
                    examples_rep_mean = torch.mean(examples_rep, dim=1)

                if args.hint_retriever:
                    # retrieve the hint representation of the closest concept
                    closest_neighbor_idx = gen_retriever(args.hint_retriever)(examples_rep_mean, hint_rep_dict[0]) 

                    # calculating retrival accuracy
                    raw_scores = torch.prod(torch.eq(hint_rep_dict[1][closest_neighbor_idx], hint_seq.cuda()).float(), dim=1)
                    retrival_acc = torch.mean(raw_scores)
                    retrival_acc_meter.update(retrival_acc, batch_size, raw_scores=(raw_scores))

                if args.poe:
                    # Compute support image -> query image scores
                    image_score = scorer_model.score(examples_rep_mean,
                                                     image_rep)

                if args.infer_hyp:
                    # Hypothesize text from examples
                    # Pick the best caption based on how well it describes concepts
                    best_predictions = np.zeros(batch_size, dtype=np.uint8)
                    best_hint_scores = np.full(batch_size,
                                               -np.inf,
                                               dtype=np.float32)

                    support_hint = hint_seq
                    for j in range(args.n_infer):
                        # Decode greedily for first hyp; otherwise sample
                        # If --oracle, hint_seq/hint_length is given
                        if args.hint_retriever:
                            hint_seq = hint_rep_dict[1][closest_neighbor_idx]
                            hint_length = hint_rep_dict[2][closest_neighbor_idx]
                        elif not args.oracle:
                            hint_seq, hint_length = proposal_model.sample(
                                examples_rep_mean,
                                sos_index,
                                eos_index,
                                pad_index,
                                greedy=j == 0)
                        elif args.oracle:
                            pass
                        else:
                            raise RuntimeError("Should not reach here")
                        
                        # Only generate hint if not using retrieval 
                        hint_seq = hint_seq.to(device)
                        hint_length = hint_length.to(device)
                        hint_rep = hint_model(hint_seq, hint_length)

                        # Compute how well this hint describes the 4 concepts.
                        if not args.oracle:
                            hint_scores = scorer_model.batchwise_score(
                                hint_rep, examples_rep)
                            hint_scores = hint_scores.cpu().numpy()

                        # Compute prediction for this hint
                        if args.multimodal_concept:
                            hint_rep = multimodal_model(
                                hint_rep, examples_rep_mean)
                        score = scorer_model.score(hint_rep, image_rep)

                        if args.poe:
                            # Average with image score
                            score = score + image_score
                        label_hat = score > 0
                        label_hat = label_hat.cpu().numpy()

                        # Update scores and predictions for best running hints
                        if not args.oracle and not args.hint_retriever:
                            updates = hint_scores > best_hint_scores
                            best_hint_scores = np.where(
                                updates, hint_scores, best_hint_scores)
                            best_predictions = np.where(
                                updates, label_hat, best_predictions)
                        else:
                            best_predictions = label_hat
                    hint_seq = idx2word(hint_seq, data_loader.dataset.i2w, remove_pad=True)
                    support_hint = idx2word(support_hint, data_loader.dataset.i2w, remove_pad=True, target=True)
                    bleu_n4 = bleu_score(hint_seq, support_hint, max_n=4, weights=[0.0, 0.0, 0.0, 1.0])
                    bleu_meter_n4.update(bleu_n4, batch_size, raw_scores=[bleu_n4])
                    bleu_n3 = bleu_score(hint_seq, support_hint,  max_n=3, weights=[0.0, 0.0, 1.0])
                    bleu_meter_n3.update(bleu_n3, batch_size, raw_scores=[bleu_n3])
                    bleu_n2 = bleu_score(hint_seq, support_hint, max_n=2, weights=[0, 1.0])
                    bleu_meter_n2.update(bleu_n2, batch_size, raw_scores=[bleu_n2])
                    bleu_n1 = bleu_score(hint_seq, support_hint, max_n=1, weights=[1.0])
                    bleu_meter_n1.update(bleu_n1, batch_size, raw_scores=[bleu_n1])
                    accuracy = accuracy_score(label_np, best_predictions)
                    precision = precision_score(label_np, best_predictions)
                    recall = recall_score(label_np, best_predictions)
                else:
                    # Compare image directly to example rep
                    score = scorer_model.score(examples_rep_mean, image_rep)

                    label_hat = score > 0
                    label_hat = label_hat.cpu().numpy()

                    accuracy = accuracy_score(label_np, label_hat)
                    precision = precision_score(label_np, label_hat)
                    recall = recall_score(label_np, label_hat)

                accuracy_meter.update(accuracy,
                                      batch_size,
                                      raw_scores=(label_hat == label_np))
                precision_meter.update(precision,
                                      batch_size,
                                      raw_scores=[precision])
                recall_meter.update(recall,
                                      batch_size,
                                      raw_scores=[recall])

        print('====> {:>12}\tEpoch: {:>3}\tAccuracy: {:.4f}\tPrecision: {:.4f}\tRecall: {:.4f}\
            \tBLEU_n1 Score: {:.4f}\tBLEU_n2 Score: {:.4f} \tBLEU_n3 Score: {:.4f}\tBLEU_n4 Score: {:.4f}\tRetrieval Accuracy: {:.4f}'.format(
            '({})'.format(split), epoch, accuracy_meter.avg, precision_meter.avg, recall_meter.avg, \
                bleu_meter_n1.avg, bleu_meter_n2.avg, bleu_meter_n3.avg, bleu_meter_n4.avg, retrival_acc_meter.avg))
        return accuracy_meter.avg, accuracy_meter.raw_scores, precision_meter.avg, recall_meter.avg, \
            bleu_meter_n1.avg, bleu_meter_n2.avg, bleu_meter_n3.avg, bleu_meter_n4.avg

    tre_comp_fn = TRE_COMP_FNS[args.tre_comp]()
    tre_err_fn = TRE_ERR_FNS[args.tre_err]()

    def eval_tre(epoch, split='train'):
        image_model.eval()
        scorer_model.eval()
        if args.infer_hyp:
            # If predicting hyp only, ignore encode/decode models for eval
            proposal_model.eval()
            hint_model.eval()
            if args.multimodal_concept:
                multimodal_model.eval()

        data_loader = data_loader_dict[split]

        all_reps = []
        all_feats = []
        with torch.no_grad():
            for examples, image, label, hint_seq, hint_length, *rest in data_loader:
                examples = examples.to(device)
                image = image.to(device)
                label = label.to(device)
                batch_size = len(image)
                n_examples = examples.shape[1]

                # Extract hint features
                hint_text = data_loader.dataset.to_text(hint_seq)
                hint_feats = [tuple(e) for e in extract_features(hint_text)]
                # TODO: Enable eval by concept vs img
                # Extend x4
                hint_feats = [h for h in hint_feats for _ in range(4)]
                all_feats.extend(hint_feats)

                # Learn image reps
                examples_2d = examples.view(batch_size * n_examples,
                                            *examples.shape[2:])
                examples_2d_rep = image_model(examples_2d)
                all_reps.append(examples_2d_rep)
        # Combine representations
        all_reps = torch.cat(all_reps, 0)
        all_feats, all_feats_mask, vocab = combine_feats(all_feats)
        all_feats = all_feats.to(all_reps.device)
        all_feats_mask = all_feats_mask.to(all_reps.device)
        tres = tre(all_reps,
                   all_feats,
                   all_feats_mask,
                   vocab,
                   tre_comp_fn,
                   tre_err_fn,
                   quiet=True)
        tres_mean = np.mean(tres)
        tres_std = np.std(tres)
        print('====> {:>12}\tEpoch: {:>3}\tTRE: {:.4f} ± {:.4f}'.format(
            '({})'.format(split), epoch, tres_mean,
            1.96 * tres_std / np.sqrt(len(tres))))
        return np.mean(tres), np.std(tres)

    best_epoch = 0
    best_epoch_acc = 0
    best_val_acc = 0
    best_val_same_acc = 0
    best_val_tre = 0
    best_val_tre_std = 0
    best_test_acc = 0
    best_test_same_acc = 0
    best_test_acc_ci = 0
    lowest_val_tre = 1e10
    lowest_val_tre_std = 0
    metrics = defaultdict(lambda: [])

    val_acc_collection = []
    bleu_n1_collection = []
    bleu_n2_collection = []
    bleu_n3_collection = []
    bleu_n4_collection = []

    save_defaultdict_to_fs(vars(args), os.path.join(args.exp_dir, 'args.json'))
    hint_rep_dict = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        if epoch % 10 != 1:
            continue
        # storing seen concepts' hint representations
        if args.hint_retriever:
            train_dataset.augment = False # this is not gonna work if there are multiple workers
            hint_rep_dict = construct_dict(train_loader, image_model, hint_model)
            train_dataset.augment = True
        train_acc, _, train_prec, train_reca, *_ = test(epoch, 'train', hint_rep_dict)
        val_acc, _, val_prec, val_reca, *_ = test(epoch, 'val', hint_rep_dict)
        # Evaluate tre on validation set
        #  val_tre, val_tre_std = eval_tre(epoch, 'val')
        val_tre, val_tre_std = 0.0, 0.0

        test_acc, test_raw_scores, test_prec, test_reca, \
            test_bleu_n1, test_bleu_n2, test_bleu_n3, test_bleu_n4 = test(epoch, 'test', hint_rep_dict)
        if has_same:
            val_same_acc, _, val_same_prec, val_same_reca, *_ = test(epoch, 'val_same', hint_rep_dict)
            test_same_acc, test_same_raw_scores, test_same_prec, test_same_reca,\
                test_same_bleu_n1, test_same_bleu_n2, test_same_bleu_n3, test_same_bleu_n4 = test(epoch, 'test_same', hint_rep_dict)    
            all_test_raw_scores = test_raw_scores + test_same_raw_scores
        else:
            val_same_acc = val_acc
            test_same_acc = test_acc
            all_test_raw_scores = test_raw_scores

        wandb.log({"loss": train_loss, 'train_acc': train_acc, 'train_prec': train_prec, 'train_reca': train_reca,\
            'val_same_acc': val_same_acc, 'val_same_prec': val_same_prec, 'val_same_reca': val_same_reca,\
            'val_acc': val_acc,'val_prec': val_prec, 'val_reca': val_reca,\
            'test_same_acc': test_same_acc, 'test_same_prec': test_same_prec, 'test_same_reca': test_same_reca,\
            'test_acc': test_acc, 'test_prec': test_prec, 'test_reca': test_reca})
        
        # Compute confidence intervals
        n_test = len(all_test_raw_scores)
        test_acc_ci = 1.96 * np.std(all_test_raw_scores) / np.sqrt(n_test)

        epoch_acc = (val_acc + val_same_acc) / 2
        is_best_epoch = epoch_acc > (best_val_acc + best_val_same_acc) / 2
        average_bleu_n1 = (test_same_bleu_n1 + test_bleu_n1) / 2
        average_bleu_n2 = (test_same_bleu_n2 + test_bleu_n2) / 2
        average_bleu_n3 = (test_same_bleu_n3 + test_bleu_n3) / 2
        average_bleu_n4 = (test_same_bleu_n4 + test_bleu_n4) / 2
        val_acc_collection.append(epoch_acc)
        bleu_n1_collection.append(average_bleu_n1)
        bleu_n4_collection.append(average_bleu_n4)
        bleu_n2_collection.append(average_bleu_n2)
        bleu_n3_collection.append(average_bleu_n3)

        if is_best_epoch:
            best_epoch = epoch
            best_epoch_acc = epoch_acc
            best_val_acc = val_acc
            best_val_same_acc = val_same_acc
            best_val_tre = val_tre
            best_val_tre_std = val_tre_std
            best_test_acc = test_acc
            best_test_same_acc = test_same_acc
            best_test_acc_ci = test_acc_ci
            best_test_bleu_n1 = average_bleu_n1
            best_test_bleu_n2 = average_bleu_n2
            best_test_bleu_n3 = average_bleu_n3
            best_test_bleu_n4 = average_bleu_n4
        if val_tre < lowest_val_tre:
            lowest_val_tre = val_tre
            lowest_val_tre_std = val_tre_std

        if args.save_checkpoint:
            raise NotImplementedError

        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)
        metrics['val_same_acc'].append(val_same_acc)
        metrics['val_tre'].append(val_tre)
        metrics['val_tre_std'].append(val_tre_std)
        metrics['test_acc'].append(test_acc)
        metrics['test_same_acc'].append(test_same_acc)
        metrics['test_acc_ci'].append(test_acc_ci)

        metrics = dict(metrics)
        # Assign best accs
        metrics['best_epoch'] = best_epoch
        metrics['best_val_acc'] = best_val_acc
        metrics['best_val_same_acc'] = best_val_same_acc
        metrics['best_val_tre'] = best_val_tre
        metrics['best_val_tre_std'] = best_val_tre_std
        metrics['best_test_acc'] = best_test_acc
        metrics['best_test_same_acc'] = best_test_same_acc
        metrics['best_test_acc_ci'] = best_test_acc_ci
        metrics['lowest_val_tre'] = lowest_val_tre
        metrics['lowest_val_tre_std'] = lowest_val_tre_std
        metrics['has_same'] = has_same
        save_defaultdict_to_fs(metrics,
                               os.path.join(args.exp_dir, 'metrics.json'))

    print('====> DONE')
    print('====> BEST EPOCH: {}'.format(best_epoch))
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_val)', best_epoch, best_val_acc))
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_val_same)', best_epoch, best_val_same_acc))
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}\tCI: {:.4f}'.format(
        '(best_test)', best_epoch, best_test_acc, best_test_acc_ci))
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_test_same)', best_epoch, best_test_same_acc))
    print('====>')
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_val_avg)', best_epoch, (best_val_acc + best_val_same_acc) / 2))
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_test_avg)', best_epoch,
        (best_test_acc + best_test_same_acc) / 2))
    print('====> {:>17}\tEpoch: {}\tAccuracy CI: {:.4f}'.format(
        '(best_test_acc_ci)', best_epoch,
        best_test_acc_ci))
    print('====> {:>17}\tEpoch: {}\tBLEU_N1: {:.4f}'.format(
        '(best_test_bleu_n1)', best_epoch,
        best_test_bleu_n1))
    print('====> {:>17}\tEpoch: {}\tBLEU_N1: {:.4f}'.format(
        '(best_test_bleu_n2)', best_epoch,
        best_test_bleu_n2))
    print('====> {:>17}\tEpoch: {}\tBLEU_N4: {:.4f}'.format(
        '(best_test_bleu_n3)', best_epoch,
        best_test_bleu_n3))
    print('====> {:>17}\tEpoch: {}\tBLEU_N4: {:.4f}'.format(
        '(best_test_bleu_n4)', best_epoch,
        best_test_bleu_n4))
    if args.plot_bleu_score:
        x = (np.array(range(len(val_acc_collection))) + 1)
        plt.plot(x, val_acc_collection, label = "validation accuracy")
        plt.plot(x, bleu_n1_collection, label = "bleu n=1")
        plt.plot(x, bleu_n2_collection, label = "bleu n=2")
        plt.plot(x, bleu_n3_collection, label = "bleu n=3")
        plt.plot(x, bleu_n4_collection, label = "bleu n=4")
        plt.xlabel('epoch')
        plt.ylabel('%')
        plt.legend( loc="upper right")
        plt.savefig('accuracy_vs_bleu_original.png')

