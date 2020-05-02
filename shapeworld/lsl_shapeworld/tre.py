import torch
from torch import nn
from torch import optim
from tqdm import trange
from torch.nn.modules.distance import CosineSimilarity


def flatten(l):
    if not isinstance(l, tuple):
        return (l, )

    out = ()
    for ll in l:
        out = out + flatten(ll)
    return out


class L1Dist(nn.Module):
    def forward(self, pred, target):
        return torch.norm(pred - target, p=1, dim=1)


class L2Dist(nn.Module):
    def forward(self, pred, target):
        return torch.norm(pred - target, p=2, dim=1)


class CosDist(nn.Module):
    def __init__(self):
        super().__init__()
        self.cossim = CosineSimilarity()

    def forward(self, x, y):
        return 1 - self.cossim(x, y)


class AddComp(nn.Module):
    def forward(self, embs, embs_mask):
        """
        embs: (batch_size, max_feats, h)
        embs_mask: (batch_size, max_feats)
        """
        embs_mask_exp = embs_mask.float().unsqueeze(2).expand_as(embs)
        embs_zeroed = embs * embs_mask_exp
        composed = embs_zeroed.sum(1)
        return composed


class MulComp(nn.Module):
    def forward(self, embs, embs_mask):
        """
        embs: (batch_size, max_feats, h)
        embs_mask: (batch_size, max_feats)
        """
        raise NotImplementedError


class Objective(nn.Module):
    def __init__(self, vocab, repr_size, comp_fn, err_fn, zero_init):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), repr_size)
        if zero_init:
            self.emb.weight.data.zero_()
        self.comp = comp_fn
        self.err = err_fn

    def compose(self, feats, feats_mask):
        """
        Input:
        batch_size, max_feats
        Output:
        batch_size, h
        """
        embs = self.emb(feats)
        # Compose embeddings
        composed = self.comp(embs, feats_mask)
        return composed

    def forward(self, rep, feats, feats_mask):
        return self.err(self.compose(feats, feats_mask), rep)


def tre(reps,
        feats,
        feats_mask,
        vocab,
        comp_fn,
        err_fn,
        quiet=False,
        steps=400,
        include_pred=False,
        zero_init=True):

    obj = Objective(vocab, reps.shape[1], comp_fn, err_fn, zero_init)
    obj = obj.to(reps.device)
    opt = optim.Adam(obj.parameters(), lr=0.001)

    if not quiet:
        ranger = trange(steps, desc='TRE')
    else:
        ranger = range(steps)
    for t in ranger:
        opt.zero_grad()
        loss = obj(reps, feats, feats_mask)
        total_loss = loss.sum()
        total_loss.backward()
        if not quiet and t % 100 == 0:
            print(total_loss.item())
        opt.step()

    final_losses = [l.item() for l in loss]
    if include_pred:
        lexicon = {
            k: obj.emb(torch.LongTensor([v])).data.cpu().numpy()
            for k, v in vocab.items()
        }
        composed = [obj.compose(f, fm) for f, fm in zip(feats, feats_mask)]
        return final_losses, lexicon, composed
    else:
        return final_losses
