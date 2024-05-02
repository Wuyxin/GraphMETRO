import torch
import torch.nn.functional as F


def compute_loss(y_pred: torch.FloatTensor, 
                 y_true: torch.LongTensor):
    if len(y_pred.size()) > 1 and  y_pred.size(-1) > 1:
        return F.cross_entropy(y_pred, y_true)
    else:
        return F.binary_cross_entropy_with_logits(y_pred.view(-1), 
                                                  y_true.view(-1).float())
    
    
