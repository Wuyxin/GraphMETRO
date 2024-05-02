import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from graphmetro.models.layers import GeneralLayer
from graphmetro.models.layers import global_add_pool
from graphmetro.config import cfg


class GNNEncoder(nn.Module):
    def __init__(self, dim_inner, dim_out, 
                 dim_in=-1, has_glob_pool=False, **kwargs):
        super(GNNEncoder, self).__init__()
        
        self.node_lin = GeneralLayer(
            'linear', dim_in=dim_in, dim_out=dim_inner
            )

        if cfg.gnn.layers_mp > 0:
            for i in range(1, cfg.gnn.layers_mp + 1):
                layer = GeneralLayer(cfg.gnn.layer_type, 
                                     dim_in=dim_inner, 
                                     dim_out=dim_inner, 
                                     has_act=True,
                                     has_bn=cfg.gnn.batchnorm,
                                     **kwargs)
                self.add_module('Layer_{}'.format(i), layer)
        if cfg.dataset.task == 'graph' or has_glob_pool:
            self.global_pooling = global_add_pool
        else:
            self.global_pooling = None
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for name, module in self.named_children():
            x = module(x, edge_index, edge_attr, batch)
        if self.global_pooling is not None:
            x = self.global_pooling(x, batch)
        return x
        

class GNN(nn.Module):
    """
    General GNN model: encoder + stage + head
    Args:
        dim_inner (int): Inner dimension
        dim_out (int): Output dimension
        dim_in (int): Input dimension
        Metadata (dict): metadata of heterogenous data object
        **kwargs (optional): Optional additional args
    """
    def __init__(self, dim_inner, dim_out, 
                 dim_in=-1, has_glob_pool=False, has_post_mp=True, **kwargs):
        super(GNN, self).__init__()

        self.encoder = GNNEncoder(dim_inner, dim_out, 
                                  dim_in=-1, has_glob_pool=False, **kwargs)
        if has_post_mp:
            self.classifier = GeneralLayer(
                'linear', dim_in=dim_inner, dim_out=dim_out, has_bn=False)
        else:
            self.classifier = None
        self.apply(init_weights)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.encoder(x, edge_index, edge_attr, batch)
        x = self.classifier(x, edge_index, edge_attr, batch)
        return x

    
def init_weights(m):
    r"""
    Performs weight initialization
    Args:
        m (nn.Module): PyTorch module
    """
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data = nn.init.xavier_uniform_(
            m.weight.data, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            m.bias.data.zero_()
            