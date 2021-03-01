"""
Training script
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
from datasets import ShapeWorld, extract_features
from datasets import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from models import ImageRep, TextRep, TextProposal, ExWrapper
from models import MultimodalRep
from models import DotPScorer, BilinearScorer
from vision import Conv4NP, ResNet18
from tre import AddComp, MulComp, CosDist, L1Dist, L2Dist, tre
from retrivers import dot_product, cos_similarity, l2_distance

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, help='Output directory')
    hyp_prediction = parser.add_mutually_exclusive_group()
    hyp_prediction.add_argument(
        '--predict_concept_hyp',
        action='store_true',
        help='Predict concept hypotheses during training')
    hyp_prediction.add_argument(
        '--predict_image_hyp',
        action='store_true',
        help='Predict image hypotheses during training')
    hyp_prediction.add_argument('--infer_hyp',
                                action='store_true',
                                help='Use hypotheses for prediction')
    parser.add_argument('--backbone',
                        choices=['vgg16_fixed', 'conv4', 'resnet18'],
                        default='vgg16_fixed',
                        help='Image model')
    parser.add_argument(
        '--multimodal_concept',
        action='store_true',
        help='Concept is a combination of hypothesis + image rep')
    parser.add_argument('--comparison',
                        choices=['dotp', 'bilinear'],
                        default='dotp',
                        help='How to compare support to query reps')
    parser.add_argument('--dropout',
                        default=0.0,
                        type=float,
                        help='Apply dropout to comparison layer')
    parser.add_argument('--debug_bilinear',
                        action='store_true',
                        help='If using bilinear term, use identity matrix')
    parser.add_argument(
        '--poe',
        action='store_true',
        help='Product of experts: support lang -> query img '
             'x support img -> query img'
    )
    parser.add_argument('--predict_hyp_task',
                        default='generate',
                        choices=['generate', 'embed'],
                        help='hyp prediction task')
    parser.add_argument('--n_infer',
                        type=int,
                        default=10,
                        help='Number of hypotheses to infer')
    parser.add_argument(
        '--oracle',
        action='store_true',
        help='Use oracle hypotheses for prediction (requires --infer_hyp)')
    parser.add_argument(
        '--retrive_hint',
        action='store_true',
        help='use the hint of tasks seen during training time (requires --infer_hyp)')
    parser.add_argument('--max_train',
                        type=int,
                        default=None,
                        help='Max number of training examples')
    parser.add_argument('--noise',
                        type=float,
                        default=0.0,
                        help='Amount of noise to add to each example')
    parser.add_argument(
        '--class_noise_weight',
        type=float,
        default=0.0,
        help='How much of that noise should be class diagnostic?')
    parser.add_argument('--noise_at_test',
                        action='store_true',
                        help='Add instance-level noise at test time')
    parser.add_argument('--noise_type',
                        default='gaussian',
                        choices=['gaussian', 'uniform'],
                        help='Type of noise')
    parser.add_argument(
        '--fixed_noise_colors',
        default=None,
        type=int,
        help='Fix noise based on class, with a max of this many')
    parser.add_argument(
        '--fixed_noise_colors_max_rgb',
        default=0.2,
        type=float,
        help='Maximum color value a single color channel '
             'can have for noise background'
    )
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
    parser.add_argument('--tre_err',
                        default='cos',
                        choices=['cos', 'l1', 'l2'],
                        help='TRE Error Metric')
    parser.add_argument('--tre_comp',
                        default='add',
                        choices=['add', 'mul'],
                        help='TRE Composition Function')
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
    parser.add_argument('--log_interval',
                        type=int,
                        default=10,
                        help='How often to log loss')
    parser.add_argument('--pred_lambda',
                        type=float,
                        default=1.0,
                        help='Weight on prediction loss')
    parser.add_argument('--hypo_lambda',
                        type=float,
                        default=10.0,
                        help='Weight on hypothesis hypothesis')
    parser.add_argument('--save_checkpoint',
                        action='store_true',
                        help='Save model')
    parser.add_argument('--cuda',
                        action='store_true',
                        help='Enables CUDA training')
    args = parser.parse_args()

    if args.oracle and not args.infer_hyp:
        parser.error("Must specify --infer_hyp to use --oracle")

    if args.retrive_hint and not args.infer_hyp:
        parser.error("Must specify --infer_hyp to use --retrive_hint")

    if args.multimodal_concept and not args.infer_hyp:
        parser.error("Must specify --infer_hyp to use --multimodal_concept")

    if args.poe and not args.infer_hyp:
        parser.error("Must specify --infer_hyp to use --poe")

    if args.dropout > 0.0 and args.comparison == 'dotp':
        raise NotImplementedError

    args.predict_hyp = args.predict_concept_hyp or args.predict_image_hyp
    args.use_hyp = args.predict_hyp or args.infer_hyp
    args.encode_hyp = args.infer_hyp or (args.predict_hyp and args.predict_hyp_task == 'embed')
    args.decode_hyp = args.infer_hyp or (args.predict_hyp and args.predict_hyp_task == 'generate')

    if args.oracle or args.retrive_hint: 
        args.n_infer = 1  # No need to repeatedly infer, hint is given

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
    else:
        raise NotImplementedError(args.backbone)

    image_model = ExWrapper(ImageRep(backbone_model))
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
        proposal_model = TextProposal(embedding_model)
        proposal_model = proposal_model.to(device)
        params_to_optimize.extend(proposal_model.parameters())

    if args.encode_hyp:
        hint_model = TextRep(embedding_model)
        hint_model = hint_model.to(device)
        params_to_optimize.extend(hint_model.parameters())

    if args.multimodal_concept:
        multimodal_model = MultimodalRep()
        multimodal_model = multimodal_model.to(device)
        params_to_optimize.extend(multimodal_model.parameters())

    optfunc = {
        'adam': optim.Adam,
        'rmsprop': optim.RMSprop,
        'sgd': optim.SGD
    }[args.optimizer]
    optimizer = optfunc(params_to_optimize, lr=args.lr)

    def train(epoch, n_steps=100, hint_rep_dict=None):
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
                hint_rep = hint_model(hint_seq, hint_length)
                
                # storing seen concepts' hint representations
                if args.retrive_hint:
                    if len(hint_rep_dict) == 0:
                        hint_rep_dict.extend([examples_rep_mean, hint_rep, hint_seq])
                    else:
                        hint_rep_dict[0] = torch.cat((hint_rep_dict[0], examples_rep_mean), dim=0)
                        hint_rep_dict[1] = torch.cat((hint_rep_dict[1], hint_rep), dim=0)
                        hint_rep_dict[2] = torch.cat((hint_rep_dict[2], hint_seq), dim=0)
                
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
        retrival_acc_meter = AverageMeter(raw=True)
        data_loader = data_loader_dict[split]

        with torch.no_grad():
            for examples, image, label, hint_seq, hint_length, *rest in data_loader:
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

                if args.retrive_hint:
                    # retrive the hint representation of the closest concept
                    # closest_neighbor_idx = cos_similarity(examples_rep_mean, hint_rep_dict[0])
                    # closest_neighbor_idx = l2_distance(examples_rep_mean, hint_rep_dict[0]) 
                    closest_neighbor_idx = dot_product(examples_rep_mean, hint_rep_dict[0])
                    hint_rep = hint_rep_dict[1][closest_neighbor_idx]

                    # calculating retrival accuracy
                    raw_scores = torch.eq(hint_rep_dict[2][closest_neighbor_idx], hint_seq.cuda())
                    retrival_acc = torch.mean(torch.prod(raw_scores.float(), dim=1))
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

                    for j in range(args.n_infer):
                        # Decode greedily for first hyp; otherwise sample
                        # If --oracle, hint_seq/hint_length is given
                        if not args.oracle and not args.retrive_hint:
                            hint_seq, hint_length = proposal_model.sample(
                                examples_rep_mean,
                                sos_index,
                                eos_index,
                                pad_index,
                                greedy=j == 0)
                        
                        # Directly used seen representations if using retrive_hint mode
                        if not args.retrive_hint: 
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
                        if not args.oracle and not args.retrive_hint:
                            updates = hint_scores > best_hint_scores
                            best_hint_scores = np.where(
                                updates, hint_scores, best_hint_scores)
                            best_predictions = np.where(
                                updates, label_hat, best_predictions)
                        else:
                            best_predictions = label_hat

                    accuracy = accuracy_score(label_np, best_predictions)
                else:
                    # Compare image directly to example rep
                    score = scorer_model.score(examples_rep_mean, image_rep)

                    label_hat = score > 0
                    label_hat = label_hat.cpu().numpy()

                    accuracy = accuracy_score(label_np, label_hat)
                accuracy_meter.update(accuracy,
                                      batch_size,
                                      raw_scores=(label_hat == label_np))

        print('====> {:>12}\tEpoch: {:>3}\tAccuracy: {:.4f}\tRetrieval Accuracy: {:.4f}'.format(
            '({})'.format(split), epoch, accuracy_meter.avg, retrival_acc_meter.avg))

        return accuracy_meter.avg, accuracy_meter.raw_scores

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

    save_defaultdict_to_fs(vars(args), os.path.join(args.exp_dir, 'args.json'))
    for epoch in range(1, args.epochs + 1):
        hint_rep_dict = []
        train_loss = train(epoch, hint_rep_dict=hint_rep_dict)
        train_acc, _ = test(epoch, 'train', hint_rep_dict)
        val_acc, _ = test(epoch, 'val', hint_rep_dict)
        # Evaluate tre on validation set
        #  val_tre, val_tre_std = eval_tre(epoch, 'val')
        val_tre, val_tre_std = 0.0, 0.0

        test_acc, test_raw_scores = test(epoch, 'test', hint_rep_dict)
        if has_same:
            val_same_acc, _ = test(epoch, 'val_same', hint_rep_dict)
            test_same_acc, test_same_raw_scores = test(epoch, 'test_same', hint_rep_dict)
            all_test_raw_scores = test_raw_scores + test_same_raw_scores
        else:
            val_same_acc = val_acc
            test_same_acc = test_acc
            all_test_raw_scores = test_raw_scores

        # Compute confidence intervals
        n_test = len(all_test_raw_scores)
        test_acc_ci = 1.96 * np.std(all_test_raw_scores) / np.sqrt(n_test)

        epoch_acc = (val_acc + val_same_acc) / 2
        is_best_epoch = epoch_acc > (best_val_acc + best_val_same_acc) / 2
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
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_test)', best_epoch, best_test_acc))
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_test_same)', best_epoch, best_test_same_acc))
    print('====>')
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_val_avg)', best_epoch, (best_val_acc + best_val_same_acc) / 2))
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_test_avg)', best_epoch,
        (best_test_acc + best_test_same_acc) / 2))
