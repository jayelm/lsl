import torch

def dot_product(query, key):
    return torch.argmax(query @ key.T, dim=1)

def l2_distance(query, key):
    return torch.argmin(torch.cdist(query, key, 2), dim=1)

def cos_similarity(query, key):
    numerator = query @ key.T
    denominator = torch.cdist(query, key, 2)   
    return torch.argmax(numerator/denominator, dim=1)