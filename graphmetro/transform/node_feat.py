import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import mask_feature


class MaskNodeFeat(nn.Module):
    def __init__(self, p=0.05, fill_value='mean'):
        super().__init__()
        self.p = p
        self.fill_value = fill_value

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if self.fill_value == 'zero':
            fill_value = 0.0
        else:
            fill_value = getattr(torch, self.fill_value)(x)
        x = mask_feature(x, p=self.p, fill_value=fill_value)
        return x, edge_index, edge_attr, batch, None
    
    
class NoisyNodeFeat(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if x.size(0) > 1:
            std = x.float().std(dim=0)
        else:
            std = torch.ones(x.size(-1))
        x = x + self.p * std * torch.randn(x.size()).to(x.device)
        return x, edge_index, edge_attr, batch, None
    

class NodeFeatShift(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = x + self.p * x.std()
        return x, edge_index, edge_attr, batch, None
