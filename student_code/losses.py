import torch


def cosine_similarity(vec1, vec2, *args, **kwargs):
    """
    Cosine similariy between vec1 and vec2
    """
    vec1_norm = torch.linalg.norm(vec1, axis=-1)
    vec2_norm = torch.linalg.norm(vec2, axis=-1)
    dot_prod = torch.sum(vec1 * vec2, axis=-1)
    sim = dot_prod / (vec1_norm * vec2_norm)
    return sim


def cosine_distance(pred, target, *args, **kwargs):
    cos_sim = cosine_similarity(pred, target)
    cos_dist = 1 - cos_sim
    cos_dist = cos_dist.mean()
    return cos_dist
