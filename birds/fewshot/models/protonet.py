"""
ProtoNet implementation (+ LSL, L3 modification)
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Debugging only
from PIL import Image
from tqdm import tqdm


class ProtoNet(nn.Module):
    """
    A ProtoNet (Snell et al., 2017) model, with optional add-ons for LSL and
    L3.
    """
    def __init__(
        self,
        model_func,
        n_way,
        n_support,
        lsl=False,
        l3=False,
        language_model=None,
        l3_model=None,
        l3_n_infer=False,
        lang_supervision="class",
    ):
        r"""
        Initialize a ProtoNet Model.

        :param model_func: thunk that when called returns a vision backbone
        :param n_way: how many classes per episode (i.e. the n in n-way)
        :param n_support: how many images per support set (i.e. the k in k-shot)
        :param lsl: enable langauge-shaped learning (must specifiy language_model)
        :param l3: enable L3 (must specify language_model and l3_model)
        :param language_model: the RNN decoder g_\phi, used in both LSL and L3.
        :param l3_model: the RNN encoder h_\eta, used in L3 only.
        :param l3_n_infer: how many samples to take per image during L3
            test-time inference. Can be overridden dynamically in self.correct_l3
        :param lang_supervision: supervise at class or instance level? if class
            level, average image reprs before decoding
        """
        super(ProtoNet, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1  # (change depends on input)
        self.feature = model_func()
        self.feat_dim = self.feature.final_feat_dim

        if lsl:
            assert language_model is not None

        self.xent_loss = nn.CrossEntropyLoss()
        self.nll_loss = nn.NLLLoss()

        # Language settings
        self.lsl = lsl
        self.language_model = language_model
        self.lang_supervision = lang_supervision

        # L3 settings
        self.l3 = l3
        self.l3_model = l3_model
        self.l3_n_infer = l3_n_infer

        # Bilinear terms
        if self.l3:
            self.img2lang = nn.Linear(self.feat_dim, self.l3_model.hidden_size)
        else:
            self.img2lang = None

        self.img2img = nn.Linear(self.feat_dim, self.feat_dim)

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        """
        Using the vision backbone, convert input to features.

        :param x: torch.Tensor of shape (n_way, n_support + n_query, *img_size)
            where the first `n_support` images along dim 1 are support images,
            and the rest are query images
        :param is_feature: if True, x is already a feature, so just pass it
            through (separating into support and query)

        :returns: (z_support, z_query); tensors of shape (n_way,
            {n_support,n_query}, feat_size) after passing the images through
            the feature backbone (unless is_feature=True)
        """
        x = x.cuda()
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(
                self.n_way * (self.n_support + self.n_query), *x.size()[2:]
            )
            z_all = self.feature.forward(x)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)

        z_support = z_all[:, : self.n_support]
        z_query = z_all[:, self.n_support :]

        return z_support, z_query

    def correct(self, x, return_loss=False):
        """
        Compute actual predictions for the query examples and return accuracy/n_query
        """
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        if return_loss:
            loss = F.cross_entropy(scores, torch.tensor(y_query, device=scores.device))
            return float(top1_correct), len(y_query), loss
        return float(top1_correct), len(y_query)

    def correct_l3(
        self,
        x,
        n_infer=None,
        return_loss=False,
        debug=False,
        index=None,
        x_orig=None,
        debug_dir=None,
    ):
        """
        Make predictions for L3 using test-time inference (i.e. not assuming
        the presence of language)

        :param x: torch.Tensor of inputs (n_way, n_query, *image_shape)
        :param return_loss: return the loss along with the accuracy
        :param debug: save predictions to file
        :param index: index of task (for debug)
        :param x_orig: if debug, the original, unnormalized images
        :param debug_dir: where to save debug predictions to

        :returns: (accuracy, n_query, optional loss if return_loss=True)
        """
        with torch.no_grad():
            # Embed support images, then decode from them
            z_support, z_query = self.parse_feature(x, is_feature=False)

            # Decode 1 language for each image, keep based on individual agreement;
            # then average for your final prototype
            n_support_lang = self.n_way * self.n_support
            z_support_flat = z_support.contiguous().view(
                z_support.shape[0] * z_support.shape[1], -1
            )

            best_support_lang = torch.zeros(
                n_support_lang, self.l3_model.hidden_size
            ).cuda()
            best_support_scores = torch.full((n_support_lang,), -float("inf")).cuda()

            z_query_flat = z_query.contiguous().view(
                z_query.shape[0] * z_query.shape[1], -1
            )

            if debug:
                best_text = np.empty(self.n_way, dtype=np.unicode_)

            # Sample captions for each prototype. Keep the best (highest
            # scoring) caption (proto) reprs
            if n_infer is None:
                n_infer = self.l3_n_infer

            for j in range(n_infer):
                # Sample from query examples. Pick the query example that best matches
                # Flatten tasks
                lang_samp, lang_length_samp = self.language_model.sample(
                    z_support_flat, greedy=j == 0
                )

                lang_samp = lang_samp.cuda()
                lang_length_samp = lang_length_samp.cuda()

                lang_rep_samp = self.l3_model(lang_samp, lang_length_samp)

                # Compute image-caption agreement scores (only keep best
                # language for each support image)
                support_scores = self.batchwise_score(
                    lang_rep_samp, z_support_flat, support_type="lang", query_type="img"
                )

                updates = support_scores > best_support_scores

                if debug:
                    text_samp = self.langauge_model.to_text(lang_samp)
                    best_text = np.where(updates.cpu().numpy(), text_samp, best_text)

                best_support_scores = torch.where(
                    updates, support_scores, best_support_scores
                )

                lang_updates = updates.unsqueeze(1).expand_as(best_support_lang)
                best_support_lang = torch.where(
                    lang_updates, lang_rep_samp, best_support_lang
                )

            # Average best language for each support class to form prototype
            support_protos = best_support_lang.unsqueeze(1).view(
                self.n_way, self.n_support, -1
            )
            support_protos = support_protos.mean(1)

            # Compute scores, best protos vs query
            query_scores = self.score(
                support_protos, z_query_flat, support_type="lang", query_type="img"
            )
            query_scores = F.log_softmax(query_scores, 1)

            label_hat = query_scores.argmax(1).cpu().numpy()

            if debug and index < 10:  # Save 10 example tasks in debug loop
                self.save_debug(index, x_orig, query_scores, best_text, debug_dir)

            y_query = np.repeat(range(self.n_way), self.n_query)

            top1_correct = (label_hat == y_query).sum()

            if return_loss:
                loss = F.nll_loss(
                    query_scores, torch.tensor(y_query, device=query_scores.device),
                )
                return float(top1_correct), len(y_query), loss
            return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer, args):
        """
        Run a train loop.

        :param epoch: the epoch # (used for logging)
        :param train_loader: a torch.utils.data.DataLoader generated from
            data.datamgr.SetDataManager
        :param optimizer: a torch.optim.Optimzer
        :param args: other args passed to the script

        :returns: a dictionary of metrics: train_acc, train_loss, cls_loss, and
            lang_loss if applicable
        """
        avg_loss = 0
        avg_cls_loss = 0
        avg_lang_loss = 0
        acc_all = []
        for i, (x, target, (lang, lang_length, lang_mask)) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support

            optimizer.zero_grad()

            if self.lsl or self.l3:  # Load language
                # Trim padding to max length in batch
                max_lang_length = lang_length.max()
                lang = lang[:, :, :max_lang_length]
                lang_mask = lang_mask[:, :, :max_lang_length]
                lang = lang.cuda()
                lang_length = lang_length.cuda()
                lang_mask = lang_mask.cuda()

            # ==== CLASSIFICATION LOSS ===-
            if self.l3:
                cls_loss, z_support, z_query = self.set_forward_loss_l3(
                    x, (lang, lang_length), return_z=True
                )
            else:
                cls_loss, z_support, z_query = self.set_forward_loss(x, return_z=True)
            loss = cls_loss

            # ==== LANGUAGE LOSS ====
            if self.lsl or self.l3:
                lang_loss = self.set_lang_loss(
                    z_support, z_query, lang, lang_length, lang_mask
                )
                lang_loss = args.lang_lambda * lang_loss
                loss = loss + lang_loss
                avg_lang_loss = avg_lang_loss + lang_loss.item()

            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            avg_cls_loss = avg_cls_loss + cls_loss.item()

            if self.l3:
                # Stick to just 1 inference at train time since evaluating
                # accuracy is expensive
                correct_this, count_this = self.correct_l3(x, n_infer=1)
            else:
                correct_this, count_this = self.correct(x)

            acc_all.append(correct_this / count_this * 100)

        metrics = {
            "train_acc": None,
            "train_loss": None,
            "cls_loss": None,
            "lang_loss": None,
        }
        metrics["train_loss"] = avg_loss / (i + 1)
        metrics["cls_loss"] = avg_cls_loss / (i + 1)
        tqdm.write("Epoch {:d} | Loss {:f}".format(epoch, metrics["train_loss"]))

        if self.lsl:
            metrics["lang_loss"] = avg_lang_loss / (i + 1)
            tqdm.write(
                "Epoch {:d} | Lang Loss {:f}".format(epoch, metrics["lang_loss"])
            )

        metrics["train_acc"] = np.mean(acc_all)
        tqdm.write("Epoch {:d} | Train Acc {:.2f}".format(epoch, metrics["train_acc"]))

        return metrics

    def test_loop(
        self,
        test_loader,
        verbose=False,
        normalizer=None,
        return_all=False,
        debug=False,
        debug_dir=None,
    ):
        """
        Run a model test loop

        :param test_loader: torch.utils.data.DataLoader for testing, generated
            by data.datamgr.SetDataManager
        :param verbose: if verbose, use tqdm to display progress
        :param normalizer: a torchvision.transforms.Transform object used to
            normalize the image before evaluation. Used if debug is set, and we
            want the original image to save to img file
        :param return_all: return an np.array of hits (1s or 0s), instead of
            summary loss/acc statistics
        :param debug: don't actually evaluate test loop; evaluate a few
            episodes then save their results in `debug_dir`
        :param debug_dir: if debug is set, save to this directory

        :returns: either an (acc, loss) tuple, or an np.array of 1s and 0s,
            where 1 indicates a correct prediction, for the entire dataset
        """
        acc_all = []
        loss_all = []

        iter_num = len(test_loader)
        if verbose:
            ranger = tqdm(enumerate(test_loader), desc="test", total=len(test_loader))
        else:
            ranger = enumerate(test_loader)
        for i, (x, target, lang) in ranger:

            if normalizer is not None:
                xdim = x.shape
                xflat = x.clone().view(xdim[0] * xdim[1], *xdim[2:])
                xnorm = torch.stack([normalizer(x) for x in xflat])
                xnorm = xnorm.view(*xdim)
            else:
                xnorm = x
            self.n_query = x.size(1) - self.n_support
            if self.l3:
                correct_this, count_this, loss_this = self.correct_l3(
                    xnorm,
                    return_loss=True,
                    debug=debug,
                    index=i,
                    x_orig=x,
                    debug_dir=debug_dir,
                )
            else:
                correct_this, count_this, loss_this = self.correct(
                    xnorm, return_loss=True
                )
            acc_all.append(correct_this / count_this * 100)
            loss_all.append(loss_this.item())

        acc_all = np.asarray(acc_all)
        loss_all = np.asarray(loss_all)
        acc_mean = np.mean(acc_all)
        loss_mean = np.mean(loss_all)
        acc_std = np.std(acc_all)
        tqdm.write(
            "%d Test Loss %f Acc = %4.2f%% +- %4.2f%%"
            % (iter_num, loss_mean, acc_mean, 1.96 * acc_std / np.sqrt(iter_num))
        )

        if return_all:
            return acc_all
        return acc_mean, loss_mean

    def get_scorer(self, support_type, query_type):
        """
        Get the bilinear term corresponding to the support and query types.
        Note bilinear term maps query embedding space into support embedding
        space, hence the reversed names.
        """
        if support_type == "lang" and query_type == "img":
            return self.img2lang
        elif support_type == "img" and query_type == "img":
            return self.img2img
        else:
            raise ValueError(
                f"support_type = {support_type}, query_type = {query_type}"
            )

    def score(self, support_proto, query, support_type="img", query_type="img"):
        """
        Score two representations, depending on their type (img <-> img as done
        in Meta/LSL, or lang <-> img as done in L3)
        """
        bilinear_term = self.get_scorer(support_type, query_type)
        W_query = bilinear_term(query)
        query_scores = torch.mm(support_proto, W_query.transpose(1, 0)).transpose(1, 0)
        return query_scores

    def batchwise_score(
        self, support_proto, query, support_type="img", query_type="img"
    ):
        """
        Do a batchwise 1-1 scoring of support and query images.

        If support proto S has dim (B, H) and query Q has dim (B, H), compute
        only distances between each s_i, b_i for i in [1, B]. So output has dim
        (B, ).
        """
        bilinear_term = self.get_scorer(support_type, query_type)
        W_query = bilinear_term(query)
        query_scores = torch.bmm(
            support_proto.unsqueeze(1), W_query.unsqueeze(2)
        ).squeeze()

        return query_scores

    def set_forward(self, x, is_feature=False, return_z=False):
        """
        Compute reps for support and query images and score the query images
        """
        z_support_orig, z_query_orig = self.parse_feature(x, is_feature,)

        z_support = z_support_orig
        z_query = z_query_orig

        z_support = z_support.contiguous()
        z_proto = z_support.mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        scores = self.score(z_proto, z_query, support_type="img", query_type="img")
        if return_z:
            return scores, z_support_orig, z_query_orig
        return scores

    def set_forward_loss(self, x, return_z=False):
        """
        Get the classification loss for Meta/LSL, where we compare prototypes
        to query images.
        """
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.cuda()

        if return_z:
            scores, z_support, z_query = self.set_forward(x, return_z=True,)
        else:
            scores = self.set_forward(x)

        loss = self.xent_loss(scores, y_query)
        if return_z:
            return loss, z_support, z_query
        return loss

    def set_forward_loss_l3(self, x, lang_all, return_z=False):
        """
        Get the classification loss for L3, where we compare queries to
        groundtruth language.
        """
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda()

        # Embed images
        z_support, z_query = self.parse_feature(x, is_feature=False)
        z_query_img = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # Embed support lang
        lang, lang_length = lang_all
        lang_support = (
            lang[:, : self.n_support]
            .contiguous()
            .view(lang.shape[0] * self.n_support, -1)
        )
        lang_length_support = lang_length[:, : self.n_support].contiguous().view(-1)

        z_support_lang = self.l3_model(lang_support, lang_length_support)
        z_support_lang = z_support_lang.unsqueeze(1).view(
            lang.shape[0], self.n_support, -1
        )
        # Proto repr is the mean of the language associated with the support set
        z_proto_lang = z_support_lang.mean(1)

        # lang <-> img comparison
        scores = self.score(
            z_proto_lang, z_query_img, support_type="lang", query_type="img"
        )
        scores = F.log_softmax(scores, 1)

        loss = self.nll_loss(scores, y_query)

        if return_z:
            return loss, z_support, z_query
        return loss

    def set_lang_loss(
        self, z_support, z_query, lang, lang_length, lang_mask,
    ):
        """
        Compute the language decoding loss, used in both L3 and LSL.
        """
        assert self.language_model.proj_h.in_features == z_support.shape[2]

        z = torch.cat((z_support, z_query), dim=1)
        if self.lang_supervision == "class":
            z = self.class_average(z)

        # Flatten out meta classes
        n_way = z.shape[0]
        n_total = z.shape[1]
        z = z.view(n_way * n_total, -1)
        lang = lang.view(n_way * n_total, -1)
        lang_mask = lang_mask.view(n_way * n_total, -1)
        lang_length = lang_length.view(n_way * n_total)

        hyp_batch_size = z.shape[0]
        seq_len = lang.shape[1]

        hypo = self.language_model(z, lang, lang_length)

        # Predict all tokens besides start of sentence (which is already given)
        hypo_nofinal = hypo[:, :-1].contiguous()
        lang_nostart = lang[:, 1:].contiguous()
        mask_nostart = lang_mask[:, 1:].contiguous()

        hypo_nofinal_2d = hypo_nofinal.view(hyp_batch_size * (seq_len - 1), -1)
        lang_nostart_2d = lang_nostart.long().view(hyp_batch_size * (seq_len - 1))
        hypo_loss = F.cross_entropy(hypo_nofinal_2d, lang_nostart_2d, reduction="none")
        hypo_loss = hypo_loss.view(hyp_batch_size, (seq_len - 1))
        # Mask out sequences based on length
        hypo_loss.masked_fill_(mask_nostart, 0.0)
        # Sum over timesteps / divide by length
        hypo_loss_per_sentence = torch.div(
            hypo_loss.sum(dim=1), (lang_length - 1).float()
        )
        hypo_loss = hypo_loss_per_sentence.mean()

        return hypo_loss

    def class_average(self, z):
        """
        Average representations for each class...then retile to make it the
        same shape as z
        """
        z_class_mean = z.mean(dim=1)
        z_class_tiled = z_class_mean.unsqueeze(1).expand(*z.shape)
        return z_class_tiled.contiguous()

    def save_debug(self, index, x, query_scores, text, debug_dir):
        """
        For debugging, save support and 5 query images for the task into a
        directory
        """
        task_dir = os.path.join(debug_dir, str(index))
        os.makedirs(task_dir, exist_ok=True)
        # Save sampled text
        with open(os.path.join(task_dir, 'samples.txt'), 'w') as f:
            f.write("\n".join(list(text)))
            f.write("\n")

        # Save original support images
        query_scores_normalized = F.softmax(query_scores, dim=1).cpu().numpy()
        yhat = np.argmax(query_scores_normalized, axis=1)
        gt_class = np.repeat(range(self.n_way), self.n_query)
        gt_index = np.tile(range(self.n_query), self.n_way)
        query_scores_df = {
            "p_{}".format(i): query_scores_normalized[:, i]
            for i in range(query_scores_normalized.shape[1])
        }
        query_scores_df.update(
            {
                "y": gt_class,
                "ny": gt_index,
                "n": np.arange(self.n_way * self.n_query),
                "yhat": yhat,
            }
        )
        # Check y_query and assign
        query_scores_df = pd.DataFrame(query_scores_df)
        query_scores_df.to_csv(
            os.path.join(task_dir, "query_predictions.csv"), index=False
        )
        # Save images
        x_support = x[:, : self.n_support]
        x_query = x[:, self.n_support :]
        for i in range(self.n_way):
            # Save support image
            for j in range(self.n_support):
                xij_orig = x_support[i, j].cpu().numpy()
                xij_orig = (255 * xij_orig).astype(np.uint8)
                xij_orig = np.moveaxis(xij_orig, 0, 2)
                Image.fromarray(xij_orig, mode="RGB").save(
                    os.path.join(task_dir, "support_{}_{}.png".format(i, j))
                )
            # Save 5 query images each
            for qj in range(5):
                qij_orig = x_query[i, qj].cpu().numpy()
                qij_orig = (255 * qij_orig).astype(np.uint8)
                qij_orig = np.moveaxis(qij_orig, 0, 2)
                Image.fromarray(qij_orig, mode="RGB").save(
                    os.path.join(task_dir, "query_{}_{}.png".format(i, qj))
                )
