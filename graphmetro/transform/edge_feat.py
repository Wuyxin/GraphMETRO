import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import mask_feature

    
class MaskEdgeFeat(nn.Module):
    def __init__(self, p=0.05, fill_value='mean'):
        super().__init__()
        self.p = p
        self.fill_value = fill_value

    def forward(self, x, edge_index, edge_attr, batch=None):
        if isinstance(self.fill_value, str):
            fill_value = getattr(torch, self.fill_value)(edge_attr)
        else:
            fill_value = self.fill_value
        edge_attr = mask_feature(edge_attr, p=self.p, fill_value=fill_value)
        return x, edge_index, edge_attr, batch, None
    
    
class NoisyEdgeFeat(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, x, edge_index, edge_attr, batch=None):
        edge_attr = edge_attr + torch.randn(edge_attr.size()).to(x.device) * self.p
        return x, edge_index, edge_attr, batch, None
    
    
class EdgeFeatShift(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        edge_attr = edge_attr + self.p
        return x, edge_index, edge_attr, batch, None
