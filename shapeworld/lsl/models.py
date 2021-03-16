"""
Models
"""

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils


class ExWrapper(nn.Module):
    """
    Wrap around a model and allow training on examples
    i.e. tensor inputs of shape
    (batch_size, n_ex, *img_dims)
    """

    def __init__(self, model, retrieve_mode=False):
        super(ExWrapper, self).__init__()
        self.model = model
        self.retrieve_mode = retrieve_mode

    def forward(self, x):
        batch_size = x.shape[0]
        if len(x.shape) == 5:
            n_ex = x.shape[1]
            img_dim = x.shape[2:]
            # Flatten out examples first
            x_flat = x.view(batch_size * n_ex, *img_dim)
        else:
            x_flat = x

        x_enc = self.model(x_flat)

        if len(x.shape) == 5:
            x_enc = x_enc.view(batch_size, n_ex, -1)
        
        # during training normalize across four examples
        # during testing instance normalize
        if self.retrieve_mode:
            return F.normalize(x_enc, dim=1)
        else:
            return x_enc

class Identity(nn.Module):
    def forward(self, x):
        return x


class ImageRep(nn.Module):
    r"""Two fully-connected layers to form a final image
    representation.

        VGG-16 -> FC -> ReLU -> FC

    Paper uses 512 hidden dimension.
    """

    def __init__(self, backbone=None, hidden_size=512):
        super(ImageRep, self).__init__()
        self.hidden_size = hidden_size
        if backbone is None:
            self.backbone = Identity()
            self.backbone.final_feat_dim = 4608
        else:
            self.backbone = backbone
        self.model = nn.Sequential(
            nn.Linear(self.backbone.final_feat_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)) # hidden_size

    def forward(self, x):
        x_enc = self.backbone(x)
        return self.model(x_enc)


class TextRep(nn.Module):
    r"""Deterministic Bowman et. al. model to form
    text representation.

    Again, this uses 512 hidden dimensions.
    """

    def __init__(self, embedding_module, hidden_size=512, num_layers=1, retrieve_mode=False):
        super(TextRep, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.gru = nn.GRU(self.embedding_dim, hidden_size, num_layers) # 512
        self.retrieve_mode = retrieve_mode

    def forward(self, seq, length):
        batch_size = seq.size(0)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]

        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)

        # embed your sequences
        embed_seq = self.embedding(seq)

        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist()
            if batch_size > 1 else length.data.tolist())

        _, hidden = self.gru(packed)
        hidden = hidden[-1, ...]

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]

        if self.retrieve_mode:
            return F.normalize(hidden, dim=-1)
        else:
            return hidden

class MultimodalDeepRep(nn.Module):
    def __init__(self):
        super(MultimodalDeepRep, self).__init__()
        self.model = nn.Sequential(nn.Linear(512 * 2, 512 * 2), nn.ReLU(),
                                   nn.Linear(512 * 2, 512), nn.ReLU(),
                                   nn.Linear(512, 512))

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.model(xy)


class MultimodalRep(nn.Module):
    r"""Concat Image and Text representations."""

    def __init__(self):
        super(MultimodalRep, self).__init__()
        self.model = nn.Sequential(nn.Linear(512 * 2, 512), nn.ReLU(),
                                   nn.Linear(512, 512))

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.model(xy)


class MultimodalSumExp(nn.Module):
    def forward(self, x, y):
        return x + y


class MultimodalLinearRep(nn.Module):
    def __init__(self):
        super(MultimodalLinearRep, self).__init__()
        self.model = nn.Linear(512 * 2, 512)

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.model(xy)


class MultimodalWeightedRep(nn.Module):
    def __init__(self):
        super(MultimodalWeightedRep, self).__init__()
        self.model = nn.Sequential(nn.Linear(512 * 2, 512), nn.ReLU(),
                                   nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        w = self.model(xy)
        out = w * x + (1. - w) * y
        return out


class MultimodalSingleWeightRep(nn.Module):
    def __init__(self):
        super(MultimodalSingleWeightRep, self).__init__()
        self.w = nn.Parameter(torch.normal(torch.zeros(1), 1))

    def forward(self, x, y):
        w = torch.sigmoid(self.w)
        out = w * x + (1. - w) * y
        return out


class TextProposal(nn.Module):
    r"""Reverse proposal model, estimating:

        argmax_lambda log q(w_i|x_1, y_1, ..., x_n, y_n; lambda)

    approximation to the distribution of descriptions.

    Because they use only positive labels, it actually simplifies to

        argmax_lambda log q(w_i|x_1, ..., x_4; lambda)

    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py
    """

    def __init__(self, embedding_module, hidden_size=512, num_layers=1):
        super(TextProposal, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.vocab_size = embedding_module.num_embeddings
        self.gru = nn.GRU(self.embedding_dim, hidden_size, num_layers) # 512
        self.outputs2vocab = nn.Linear(hidden_size, self.vocab_size) # 512

    def forward(self, feats, seq, length):
        # feats is from example images
        batch_size = seq.size(0)

        if batch_size > 1:
            # BUGFIX? dont we need to sort feats too?
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]
            feats = feats[sorted_idx]

        feats = feats.repeat(self.gru.num_layers,1,1) #.unsqueeze(0)
        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)

        # embed your sequences
        embed_seq = self.embedding(seq)

        packed_input = rnn_utils.pack_padded_sequence(embed_seq,
                                                      sorted_lengths.cpu())

        # shape = (seq_len, batch, hidden_dim) Visual features are used to initialize GRUs
        packed_output, _ = self.gru(packed_input, feats)
        output = rnn_utils.pad_packed_sequence(packed_output)
        output = output[0].contiguous()

        # reorder from (L,B,D) to (B,L,D)
        output = output.transpose(0, 1)

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            output = output[reversed_idx]

        max_length = output.size(1)
        output_2d = output.view(batch_size * max_length, self.hidden_size) # 512
        outputs_2d = self.outputs2vocab(output_2d)
        outputs = outputs_2d.view(batch_size, max_length, self.vocab_size)

        return outputs

    def sample(self, feats, sos_index, eos_index, pad_index, greedy=False):
        """Generate from image features using greedy search."""
        batch_size = feats.size(0)

        # initialize hidden states using image features
        states = feats.repeat(self.gru.num_layers,1,1) #.unsqueeze(0)

        # first input is SOS token
        inputs = np.array([sos_index for _ in range(batch_size)])
        inputs = torch.from_numpy(inputs)
        inputs = inputs.unsqueeze(1)
        inputs = inputs.to(feats.device)

        # save SOS as first generated token
        inputs_npy = inputs.squeeze(1).cpu().numpy()
        sampled_ids = [[w] for w in inputs_npy]

        # (B,L,D) to (L,B,D)
        inputs = inputs.transpose(0, 1)

        # compute embeddings
        inputs = self.embedding(inputs)

        for i in range(20):  # like in jacobs repo
            outputs, states = self.gru(inputs,
                                       states)  # outputs: (L=1,B,H)
            outputs = outputs.squeeze(0)  # outputs: (B,H)
            outputs = self.outputs2vocab(outputs)  # outputs: (B,V)

            if greedy:
                predicted = outputs.max(1)[1]
                predicted = predicted.unsqueeze(1)
            else:
                outputs = F.softmax(outputs, dim=1)
                predicted = torch.multinomial(outputs, 1)

            predicted_npy = predicted.squeeze(1).cpu().numpy()
            predicted_lst = predicted_npy.tolist()

            for w, so_far in zip(predicted_lst, sampled_ids):
                if so_far[-1] != eos_index:
                    so_far.append(w)

            inputs = predicted.transpose(0, 1)  # inputs: (L=1,B)
            inputs = self.embedding(inputs)  # inputs: (L=1,B,E)

        sampled_lengths = [len(text) for text in sampled_ids]
        sampled_lengths = np.array(sampled_lengths)

        max_length = max(sampled_lengths)
        padded_ids = np.ones((batch_size, max_length)) * pad_index

        for i in range(batch_size):
            padded_ids[i, :sampled_lengths[i]] = sampled_ids[i]

        sampled_lengths = torch.from_numpy(sampled_lengths).long()
        sampled_ids = torch.from_numpy(padded_ids).long()

        return sampled_ids, sampled_lengths

    def test_sample(self, feats, sos_index, eos_index, pad_index, greedy=False):
        with torch.no_grad():
            return self.sample(feats, sos_index, eos_index, pad_index, greedy=greedy)

class EmbedImageRep(nn.Module):
    def __init__(self, z_dim):
        super(EmbedImageRep, self).__init__()
        self.z_dim = z_dim
        self.model = nn.Sequential(nn.Linear(self.z_dim, 512), nn.ReLU(),
                                   nn.Linear(512, 512))

    def forward(self, x):
        return self.model(x)


class EmbedTextRep(nn.Module):
    def __init__(self, z_dim):
        super(EmbedTextRep, self).__init__()
        self.z_dim = z_dim
        self.model = nn.Sequential(nn.Linear(self.z_dim, 512), nn.ReLU(),
                                   nn.Linear(512, 512))

    def forward(self, x):
        return self.model(x)


class Scorer(nn.Module):
    def __init__(self):
        super(Scorer, self).__init__()

    def forward(self, x, y):
        raise NotImplementedError

    def score(self, x, y):
        raise NotImplementedError

    def batchwise_score(self, x, y):
        raise NotImplementedError


class DotPScorer(Scorer):
    def __init__(self):
        super(DotPScorer, self).__init__()

    def score(self, x, y):
        return torch.sum(x * y, dim=1)

    def batchwise_score(self, y, x):
        # REVERSED
        bw_scores = torch.einsum('ijk,ik->ij', (x, y))
        return torch.sum(bw_scores, dim=1)


class BilinearScorer(DotPScorer):
    def __init__(self, hidden_size, dropout=0.0, identity_debug=False):
        super(BilinearScorer, self).__init__()
        self.bilinear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout_p = dropout
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        else:
            self.dropout = lambda x: x
        if identity_debug:
            # Set this as identity matrix to make sure we get the same output
            # as DotPScorer
            self.bilinear.weight = nn.Parameter(
                torch.eye(hidden_size, dtype=torch.float32))
            self.bilinear.weight.requires_grad = False

    def score(self, x, y):
        wy = self.bilinear(y)
        wy = self.dropout(wy)
        return super(BilinearScorer, self).score(x, wy)

    def batchwise_score(self, x, y):
        """
        x: (batch_size, h)
        y: (batch_size, n_examples, h)
        """
        batch_size, n_examples, h = y.shape
        wy = self.bilinear(y.view(batch_size * n_examples,
                                  -1)).unsqueeze(1).view_as(y)
        wy = self.dropout(wy)
        # wy: (batch_size, n_examples, h)
        return super(BilinearScorer, self).batchwise_score(x, wy)
