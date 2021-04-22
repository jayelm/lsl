from transformers import LxmertConfig, LxmertModel
import torch
import torch.nn as nn
from einops import rearrange

PATCH_SIZE = 56

class Lxmert(nn.Module):

    def __init__(self, vocab_size, hidden_size, visual_feat_dim, visual_pos_dim, pretrained=False):
        super().__init__()
        
        self.pretrained = pretrained
        if self.pretrained:
            self.visual_proj = nn.Linear(visual_feat_dim, 2048)
            self.lxmert = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased')
        else:
            config = LxmertConfig(vocab_size=vocab_size, hidden_size=hidden_size, \
               visual_feat_dim=visual_feat_dim, visual_pos_dim=visual_pos_dim)
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
        
        dummy_hints = torch.tensor([[1, 0, 2] for _ in range(visual_patches.shape[0])]).reshape((visual_patches.shape[0], -1)).cuda()
        
        if visual_pos is None:
            visual_pos = self.gen_visual_pos(visual_patches.shape[0], visual_patches.shape[1])
        
        if self.pretrained:
            out = self.lxmert(dummy_hints, self.visual_proj(visual_patches), visual_pos).pooled_output
        else:
            out = self.lxmert(dummy_hints, visual_patches, visual_pos).pooled_output

        if original_shape:
            return out.reshape(*original_shape[:2], -1)
        else:
            return out
    

def init_lxmert(vocab_size, hidden_size, visual_feat_dim, visual_pos_dim):
    config = LxmertConfig(vocab_size=vocab_size, hidden_size=hidden_size, 
        visual_feat_dim=visual_feat_dim, visual_pos_dim=visual_pos_dim)
    return LxmertModel(config)

if __name__ == "__main__":
    from datasets import ShapeWorld
    from datasets import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
    from torch import optim
    from arguments import ArgumentParser
    from bertAdam import BertAdam
    
    args = ArgumentParser().parse_args()
    train_dataset = ShapeWorld(
            split='train',
            vocab=None,
            augment=True,
            precomputed_features=False,
            max_size=args.max_train,
            preprocess=True,
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

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False)

    # project_linear = nn.Linear(9408, 2048).cuda()
    lxmert = Lxmert(-1, 768, 9408, 4).cuda()
    # lxmert = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased').cuda()
    # params_to_optimize = list(project_linear.parameters())
    params_to_optimize = []
    params_to_optimize.extend(list(lxmert.parameters()))
    optimizer = BertAdam(params_to_optimize, args.lr)

    lxmert.train()
    for _ in range(args.epochs):
        total_loss = 0.0
        accuracy = []
        for batch_idx in range(100):
            examples, image, label, hint_seq, hint_length, *rest = \
                train_dataset.sample_train(args.batch_size)

            examples = examples.cuda()
            image = image.cuda()

            examples_reps_mean = lxmert(examples)
            img_rep = lxmert(image)

            score = torch.sum(examples_reps_mean * img_rep, dim=1)

            label = label.cuda()
            loss = nn.functional.binary_cross_entropy_with_logits(score, label.float())
            pred = (score > 0).int()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            accuracy.extend((pred == label).float())
        print(total_loss)
        print(torch.mean(torch.tensor(accuracy)))