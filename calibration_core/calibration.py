
import torch

def margin_confidence(probs: torch.Tensor) -> float:
    sorted_probs, _ = torch.sort(probs, descending=True)
    return float(sorted_probs[0] - sorted_probs[1])

def calibrated_decision(probs, threshold=0.2):
    margin = margin_confidence(torch.tensor(probs))
    
    if margin < threshold:
        return "LOW_CONFIDENCE"
    
    return "CONFIDENT"
