"""
Base classes for Graph Neural Networks
"""
import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch
from torch import Tensor
from graphmetro.models.good.Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool
from torch.nn import Identity
from graphmetro.config import cfg


class BasicEncoder(torch.nn.Module):
    r"""
        Base GNN feature encoder.
    """

    def __init__(self):
        super(BasicEncoder, self).__init__()
        num_layer = cfg.gnn.layers_mp

        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if not cfg.gnn.batchnorm:
            self.batch_norm1 = Identity()
            self.batch_norms = [
                Identity()
                for _ in range(num_layer - 1)
            ]
        else:
            self.batch_norm1 = nn.BatchNorm1d(cfg.gnn.dim_inner)
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(cfg.gnn.dim_inner)
                for _ in range(num_layer - 1)
            ])
        self.dropout1 = nn.Dropout(cfg.gnn.dropout)
        self.dropouts = nn.ModuleList([
            nn.Dropout(cfg.gnn.dropout)
            for _ in range(num_layer - 1)
        ])
        if cfg.dataset.task == 'node':
            self.readout = IdenticalPool()
        elif cfg.model.graph_pooling == 'mean':
            self.readout = GlobalMeanPool()
        else:
            self.readout = GlobalMaxPool()
