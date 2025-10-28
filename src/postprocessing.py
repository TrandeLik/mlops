import torch
import torch.nn.functional as F
from typing import List, Dict, Any


def convert_logits_to_predictions(
    logits: torch.Tensor, id2label: Dict[int, str]
) -> List[Dict[str, Any]]:

    if not isinstance(logits, torch.Tensor) or logits.dim() != 2:
        raise ValueError("Logits must be a 2D torch.Tensor.")
    
    probabilities = F.softmax(logits, dim=1)
    top_probs, top_indices = torch.max(probabilities, dim=1)
    
    predictions = []
    for i in range(len(top_probs)):
        pred_class_id = top_indices[i].item()
        predictions.append({
            "label": id2label[pred_class_id],
            "confidence": top_probs[i].item()
        })
    return predictions

