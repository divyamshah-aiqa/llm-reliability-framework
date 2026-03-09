import torch

def margin_confidence(probs):
    # probs = tensor of class probabilities
    top2 = torch.topk(probs, 2)
    margin = top2.values[0] - top2.values[1]
    return margin.item()
