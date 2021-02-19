"""
Representations
"""


import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils


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
        if backbone is None:
            self.backbone = Identity()
            self.backbone.final_feat_dim = 4608
        else:
            self.backbone = backbone
        self.model = nn.Sequential(
            nn.Linear(self.backbone.final_feat_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size))

    def forward(self, x):
        x_enc = self.backbone(x)
        return self.model(x_enc)


class TextRep(nn.Module):
    r"""Deterministic Bowman et. al. model to form
    text representation.

    Again, this uses 512 hidden dimensions.
    """

    def __init__(self, embedding_module):
        super(TextRep, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.gru = nn.GRU(self.embedding_dim, 512)

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

        return hidden


class MultimodalSumExp(nn.Module):
    def forward(self, x, y):
        return x + y


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


