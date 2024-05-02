import torch
from sklearn import metrics
import numpy as np
from graphmetro.config import cfg


def compute_metric(y_pred: torch.Tensor, 
                   y_true: torch.LongTensor):
    if cfg.dataset.metric == 'acc':
        return y_pred.eq(y_true).float().mean().item()
    elif cfg.dataset.metric == 'auc':
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return metrics.roc_auc_score(y_true, y_pred, multi_class='ovo')
    elif 'f1_' in cfg.dataset.metric:
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return metrics.f1_score(y_true, y_pred, 
                                average=cfg.dataset.metric[3:])
    else:
        raise NotImplementedError
    
    
