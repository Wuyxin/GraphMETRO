import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from graphmetro.config import cfg
from graphmetro.models.utils import dict_to_matrix, matrix_to_dict
from collections import OrderedDict


class GeneralLayer(nn.Module):
    '''General wrapper for layers'''
    def __init__(self,
                 name,
                 dim_in,
                 dim_out,
                 has_act=True,
                 has_bn=True,
                 has_l2norm=False,
                 **kwargs):
        super(GeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        self.layer = layer_dict[name](dim_in=dim_in,
                                      dim_out=dim_out,
                                      bias=not has_bn,
                                      **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(
                nn.BatchNorm1d(dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        if cfg.gnn.dropout > 0:
            layer_wrapper.append(
                nn.Dropout(p=cfg.gnn.dropout, inplace=cfg.mem.inplace))
        if has_act:
            layer_wrapper.append(act_dict[cfg.gnn.act]())
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, x, edge_index=None, 
                edge_attr=None, batch=None, **kwargs):
        # overwrite
        if not cfg.train.use_edge_attr:
            edge_attr = None
        else:
            assert not edge_attr is None
            if len(edge_attr.size()) == 1:
                edge_attr = edge_attr.view(-1, 1)
        x = self.layer(x, edge_index, edge_attr, batch)
        if isinstance(x, torch.Tensor):
            x = self.post_layer(x)
            if self.has_l2norm:
                x = F.normalize(x, p=2, dim=1)
        return x


class Linear(nn.Module):
    def __init__(self, dim_out, dim_in=-1, bias=False, **kwargs):
        super().__init__()
        self.model = pyg.nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        return self.model(x)
    
        
class Wrapper(torch.nn.Module):
    def __init__(self, model_1, model_2, detach=False):
        super(Wrapper, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.detach = detach
        
    def encoder(self, x, edge_index, edge_attr=None, batch=None):
        x = self.model_1(x, edge_index, edge_attr, batch)
        if self.detach:
            x = x.detach()
        return self.model_2(x)
    

class MLP(nn.Module):
    def __init__(self, dim_out, dim_in=-1, bias=False, 
                 skip_connection=False, act=None, **kwargs):
        super(MLP, self).__init__()
        
        if act is None:
            self.act = act_dict[cfg.gnn.act]()
        else:
            self.act = act
            
        dim_inner = cfg.gnn.dim_inner if dim_in == -1 else  2 * dim_in
        self.model = nn.Sequential(OrderedDict([
                ('lin1', pyg.nn.Linear(dim_in, dim_inner, bias=bias)),
                ('act', self.act),
                ('lin2', pyg.nn.Linear(dim_inner, dim_out, bias=bias))
                ]))
        self.skip_connection = skip_connection
        
    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        if self.skip_connection:
            x = x + self.model(x)
        else:
            x = self.model(x)
        return x
    
    
class GATConv(nn.Module):
    def __init__(self, dim_out, dim_in=-1, bias=False, **kwargs):
        super().__init__()
        self.model = pyg.nn.GATConv(dim_in, dim_out, 
                                    edge_dim=-1,
                                    heads=cfg.gnn.num_heads, 
                                    bias=bias, 
                                    concat=False
                                    )
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.model(x, edge_index, edge_attr)
        return x


class SAGEConv(nn.Module):
    def __init__(self, dim_out, dim_in=-1, bias=False, **kwargs):
        super().__init__()
        self.model = pyg.nn.SAGEConv(dim_in, dim_out, bias=bias)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.model(x, edge_index)
        return x
    
    
class GraphConv(nn.Module):
    def __init__(self, dim_out, dim_in=-1, bias=False, **kwargs):
        super().__init__()
        self.model = pyg.nn.GraphConv(dim_in, dim_out, bias=bias)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        assert edge_attr is None or len(edge_attr.size()) == 1 or edge_attr.size(-1) == 1
        x = self.model(x, edge_index, edge_weight=edge_attr)
        return x
    
    
class GCNConv(nn.Module):
    def __init__(self, dim_out, dim_in=-1, bias=False, **kwargs):
        super().__init__()
        self.model = pyg.nn.GCNConv(dim_in, dim_out, bias=bias)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        assert edge_attr is None or len(edge_attr.size()) == 1 or edge_attr.size(-1) == 1
        x = self.model(x, edge_index, edge_weight=edge_attr)
        return x
    
    
class GINConv(nn.Module):
    def __init__(self, dim_out, dim_in=-1, bias=False, **kwargs):
        super().__init__()
        mlp = pyg.nn.MLP([dim_in, 2 * dim_out, dim_out])
        self.model = pyg.nn.GINConv(nn=mlp, train_eps=False)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.model(x, edge_index)
        return x
    
    
act_dict = {
    'relu': nn.ReLU,
    'selu': nn.SELU,
    'prelu': nn.PReLU,
    'elu': nn.ELU,
    'lrelu_01': nn.LeakyReLU,
    'lrelu_025': nn.LeakyReLU,
    'lrelu_05': nn.LeakyReLU,
}


layer_dict = {
    'linear': Linear,
    'mlp': MLP,
    'gatconv': GATConv,
    'sageconv': SAGEConv,
    'graphconv': GraphConv,
    'gcnconv': GCNConv,
    'ginconv': GINConv
}


def global_add_pool(x, batch=None):
    return pyg.nn.global_mean_pool(x, batch) # add_pool