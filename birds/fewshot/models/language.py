"""
Language encoders/decoders.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class TextProposal(nn.Module):
    r"""Reverse proposal model, estimating:
        argmax_lambda log q(w_i|x_1, y_1, ..., x_n, y_n; lambda)
    approximation to the distribution of descriptions.
    Because they use only positive labels, it actually simplifies to
        argmax_lambda log q(w_i|x_1, ..., x_4; lambda)
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py
    """

    def __init__(
        self,
        embedding_module,
        input_size=1600,
        hidden_size=512,
        project_input=False,
        rnn="gru",
        num_layers=1,
        dropout=0.2,
        vocab=None,
        sos_index=0,
        eos_index=0,
        pad_index=0,
    ):
        super(TextProposal, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.vocab_size = embedding_module.num_embeddings
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.project_input = project_input
        self.num_layers = num_layers
        self.rnn_type = rnn
        if self.project_input:
            self.proj_h = nn.Linear(self.input_size, self.hidden_size)
            if self.rnn_type == "lstm":
                self.proj_c = nn.Linear(self.input_size, self.hidden_size)

        if rnn == "gru":
            RNN = nn.GRU
        elif rnn == "lstm":
            RNN = nn.LSTM
        else:
            raise ValueError("Unknown RNN model {}".format(rnn))

        # Init the RNN
        self.rnn = None
        self.rnn = RNN(
            self.embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)

        # Projection from RNN hidden size to output vocab
        self.outputs2vocab = nn.Linear(hidden_size, self.vocab_size)
        self.vocab = vocab
        # Get sos/eos/pad indices
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.pad_index = pad_index
        self.rev_vocab = {v: k for k, v in vocab.items()}

    def forward(self, feats, seq, length):
        # feats is from example images
        batch_size = seq.size(0)

        if self.project_input:
            feats_h = self.proj_h(feats)
            if self.rnn_type == "lstm":
                feats_c = self.proj_c(feats)
        else:
            feats_h = feats
            feats_c = feats

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]
            feats_h = feats_h[sorted_idx]
            if self.rnn_type == "lstm":
                feats_c = feats_c[sorted_idx]

        # Construct hidden states by expanding to number of layers
        feats_h = feats_h.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        if self.rnn_type == "lstm":
            feats_c = feats_c.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
            hidden = (feats_h, feats_c)
        else:
            hidden = feats_h

        # embed your sequences
        embed_seq = self.embedding(seq)

        # shape = (seq_len, batch, hidden_dim)
        packed_input = rnn_utils.pack_padded_sequence(
            embed_seq, sorted_lengths, batch_first=True
        )
        packed_output, _ = self.rnn(packed_input, hidden)
        output = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        output = output[0].contiguous()

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            output = output[reversed_idx]

        max_length = output.size(1)
        output_2d = output.view(batch_size * max_length, self.hidden_size)
        output_2d_dropout = self.dropout(output_2d)
        outputs_2d = self.outputs2vocab(output_2d_dropout)
        outputs = outputs_2d.view(batch_size, max_length, self.vocab_size)

        return outputs

    def sample(self, feats, greedy=False, to_text=False):
        """Generate from image features using greedy search."""
        with torch.no_grad():
            if self.project_input:
                feats_h = self.proj_h(feats)
                states = feats_h
                if self.rnn_type == "lstm":
                    feats_c = self.proj_c(feats)
                    states = (feats_h, feats_c)
            else:
                states = feats

            batch_size = states.size(0)

            # initialize hidden states using image features
            states = states.unsqueeze(0)

            # first input is SOS token
            inputs = np.array([self.sos_index for _ in range(batch_size)])
            inputs = torch.from_numpy(inputs)
            inputs = inputs.unsqueeze(1)
            inputs = inputs.to(feats.device)

            # save SOS as first generated token
            inputs_npy = inputs.squeeze(1).cpu().numpy()
            sampled_ids = [[w] for w in inputs_npy]

            # compute embeddings
            inputs = self.embedding(inputs)

            # Here, we use the same as max caption length
            for i in range(32):  # like in jacobs repo
                outputs, states = self.rnn(inputs, states)  # outputs: (L=1,B,H)
                outputs = outputs.squeeze(1)  # outputs: (B,H)
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
                    if so_far[-1] != self.eos_index:
                        so_far.append(w)

                inputs = predicted
                inputs = self.embedding(inputs)  # inputs: (L=1,B,E)

            sampled_lengths = [len(text) for text in sampled_ids]
            sampled_lengths = np.array(sampled_lengths)

            max_length = max(sampled_lengths)
            padded_ids = np.ones((batch_size, max_length)) * self.pad_index

            for i in range(batch_size):
                padded_ids[i, : sampled_lengths[i]] = sampled_ids[i]

            sampled_lengths = torch.from_numpy(sampled_lengths).long()
            sampled_ids = torch.from_numpy(padded_ids).long()

        if to_text:
            sampled_text = self.to_text(sampled_ids)
            return sampled_text, sampled_lengths
        return sampled_ids, sampled_lengths

    def to_text(self, sampled_ids):
        texts = []
        for sample in sampled_ids.numpy():
            texts.append(" ".join([self.rev_vocab[v] for v in sample if v != 0]))
        return np.array(texts, dtype=np.unicode_)


class TextRep(nn.Module):
    r"""Deterministic Bowman et. al. model to form
    text representation.

    Again, this uses 512 hidden dimensions.
    """

    def __init__(
        self, embedding_module, hidden_size=512, rnn="gru", num_layers=1, dropout=0.2
    ):
        super(TextRep, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        if rnn == "gru":
            RNN = nn.GRU
        elif rnn == "lstm":
            RNN = nn.LSTM
        else:
            raise ValueError("Unknown RNN model {}".format(rnn))
        self.rnn = RNN(
            self.embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_size = hidden_size

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
            sorted_lengths.data.tolist() if batch_size > 1 else length.data.tolist(),
        )

        _, hidden = self.rnn(packed)
        hidden = hidden[-1, ...]

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]

        return hidden
