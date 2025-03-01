import torch

from graphmetro.config import cfg
from graphmetro.models.gnn import GNN
from graphmetro.models.good.GCNs import GCN
from graphmetro.models.good.GINs import GIN
from graphmetro.models.good.GINvirtualnode import vGIN


def create_model(dataset, 
                 to_device=True, 
                 dim_in=None, 
                 dim_out=None, 
                 has_post_mp=True
                 ):
    r"""
    Create model for graph machine learning
    Args:
        to_device (string): The devide that the model will be transferred to
        dim_in (int, optional): Input dimension to the model
        dim_out (int, optional): Output dimension to the model
    """
    dim_inner = cfg.gnn.dim_inner if dim_in is None else dim_in
    dim_out = dataset.n_classes if dim_out is None else dim_out

    if 'GOOD' in cfg.dataset.name:
        if cfg.dataset.task == 'node':
            model = GCN(dim_inner=dim_inner, 
                        dim_out=dim_out,
                        has_post_mp=has_post_mp
                        )
        else:
            model = vGIN(dim_inner=dim_inner, 
                         dim_out=dim_out,
                         has_post_mp=has_post_mp
                         )
    else:
        model = GNN(dim_inner=dim_inner, 
                    dim_out=dim_out,
                    has_post_mp=has_post_mp
                    )
    if to_device:
        model = model.to(cfg.device)
    return model