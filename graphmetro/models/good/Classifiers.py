r"""
Applies a linear transformation to complete classification from representations.
"""
import torch
import torch.nn as nn
from torch import Tensor
from graphmetro.config import cfg


class Classifier(torch.nn.Module):
    r"""
    Applies a linear transformation to complete classification from representations.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`cfg.gnn.dim_inner`, :obj:`config.dataset.num_classes`)
    """
    def __init__(self, dim_out):

        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(*(
            [nn.Linear(cfg.gnn.dim_inner, dim_out)]
        ))

    def forward(self, x, edge_index=None, edge_attr=None, batch=None) -> Tensor:
        r"""
        Applies a linear transformation to feature representations.

        Args:
            feat (Tensor): feature representations

        Returns (Tensor):
            label predictions

        """
        return self.classifier(x)
