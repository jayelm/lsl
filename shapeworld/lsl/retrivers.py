import torch

def construct_dict(dataloader, image_model=None, hint_model=None, multimodal_model=None):
    hint_rep_dict = []
    image_model.eval()
    hint_model.eval()
    aug = dataloader.dataset.augment
    dataloader.dataset.augment = False

    for examples, image, label, hint, hint_length, *rest in dataloader:

        hint = hint.cuda()
        hint_rep = hint_model(hint, hint_length)
        examples_rep_mean = torch.mean(image_model(examples.cuda()), dim=1)
        
        if len(hint_rep_dict) == 0:
            hint_rep_dict.extend([examples_rep_mean, hint_rep, hint])
        else:
            hint_rep_dict[0] = torch.cat((hint_rep_dict[0], examples_rep_mean), dim=0)
            hint_rep_dict[1] = torch.cat((hint_rep_dict[1], hint_rep), dim=0)
            hint_rep_dict[2] = torch.cat((hint_rep_dict[2], hint), dim=0)

    dataloader.dataset.augment = aug
    print(dataloader.dataset.augment)
    return hint_rep_dict

def dot_product(query, key):
    return torch.argmax(query @ key.T, dim=1)

def l2_distance(query, key): 
    return torch.argmin(torch.cdist(query, key, 2), dim=1)

def cos_similarity(query, key):
    numerator = query @ key.T
    denominator = torch.cdist(query, key, 2)   
    return torch.argmax(numerator/denominator, dim=1)