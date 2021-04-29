from transformers import LxmertConfig, LxmertModel
import torch
import torch.nn as nn
from einops import rearrange

PATCH_SIZE = 56

class Lxmert(nn.Module):

    def __init__(self, vocab_size, hidden_size, visual_feat_dim, visual_pos_dim, initializer_range, pretrained=False):
        super().__init__()
        
        self.pretrained = pretrained
        if self.pretrained:
            self.visual_proj = nn.Linear(visual_feat_dim, 2048)
            self.lxmert = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased')
        else:
            config = LxmertConfig(vocab_size=vocab_size, hidden_size=hidden_size, \
               visual_feat_dim=visual_feat_dim, visual_pos_dim=visual_pos_dim, initializer_range=initializer_range)
            
            self.lxmert = LxmertModel(config)

    def gen_visual_pos(self, batch_size, num_visual_features):
        grid = int(num_visual_features ** (1/2))
        visual_pos = [[[i/grid, j/grid, 1/grid, 1/grid] for i in range(grid) for j in range(grid)] for _ in range(batch_size)]
        return torch.tensor(visual_pos).to(self.lxmert.device)

    def forward(self, visual_feats, visual_pos=None, input_ids=None):
        original_shape = None
        
        if len(visual_feats.shape) > 4:
            original_shape = visual_feats.shape
            visual_patches = rearrange(visual_feats, 'b k c (h p1) (w p2) -> (b k) (h w) (p1 p2 c)', p1 = PATCH_SIZE, p2 = PATCH_SIZE)
        elif len(visual_feats.shape) == 4:
            visual_patches = rearrange(visual_feats, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = PATCH_SIZE, p2 = PATCH_SIZE)
        
        if input_ids is None:
            # dummy tokens TODO: move to train
            input_ids = torch.tensor([[1, 0, 2] for _ in range(visual_patches.shape[0])]).reshape((visual_patches.shape[0], -1)).cuda()
        
        if visual_pos is None:
            visual_pos = self.gen_visual_pos(visual_patches.shape[0], visual_patches.shape[1])
        
        if self.pretrained:
            out = self.lxmert(input_ids, self.visual_proj(visual_patches), visual_pos).pooled_output
        else:
            out = self.lxmert(input_ids, visual_patches, visual_pos).pooled_output

        out = nn.functional.normalize(out, dim=-1)
        if original_shape:
            return out.reshape(*original_shape[:2], -1)
        else:
            return out
