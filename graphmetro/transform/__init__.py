from .edge_feat import MaskEdgeFeat, NoisyEdgeFeat, EdgeFeatShift
from .node_feat import MaskNodeFeat, NoisyNodeFeat, NodeFeatShift
from .topology import AddEdge, DropEdge, DropNode, DropPath, RandomSubgraph, Identity
from .utils import Compose
from graphmetro.config import cfg
import torch


def get_trans_func_dict(trans_func_names: list) -> dict: 
    
    trans_func_dict = {}
    for transform in trans_func_names:
        if transform == 'id' or len(transform.split('-')) > 1:
            trans_func_dict[transform] = transform_func(transform)
        
        elif transform == 'random_subgraph':
            if len(cfg.shift.k) == 1:
                trans_func_dict[transform] = transform_func(transform)
            else:
                for k in cfg.shift.k:
                    trans_func_dict[f'{transform}_k={k}'] = transform_func(transform, k=k)
                    
        else:
            if len(cfg.shift.p) == 1:
                trans_func_dict[transform] = transform_func(transform)
            else:
                for p in cfg.shift.p:
                    trans_func_dict[f'{transform}_p={p}'] = transform_func(transform, p=p)
        
    return trans_func_dict


def transform_func(key, p=None, k=None):
    
    if p is None:
        p = cfg.shift.p[0]
        
    if k is None:
        k = cfg.shift.k[0]
        
    transform_dict = {
        'id': Identity(),
        'mask_edge_feat': MaskEdgeFeat(p, cfg.shift.fill_value),
        'noisy_edge_feat': NoisyEdgeFeat(p),
        'edge_feat_shift': EdgeFeatShift(p),
        'mask_node_feat': MaskNodeFeat(p, cfg.shift.fill_value),
        'noisy_node_feat': NoisyNodeFeat(p),
        'node_feat_shift': NodeFeatShift(p),
        'add_edge': AddEdge(p),
        'drop_edge': DropEdge(p),
        'drop_node': DropNode(p),
        'drop_path': DropPath(p),
        'random_subgraph': RandomSubgraph(k)
    }
    if key in transform_dict.keys():
        return transform_dict[key]
    else:
        try:
            # Try composing transformation functions
            transform_names = key.split('-')
            transforms = [transform_dict[t] for t in transform_names]
            return Compose(transforms)
        except:
            raise KeyError
        
        
def get_gt_targets(key: str, known_list: list) -> torch.LongTensor:
    transform_names = key.split('-')
    # gd = []
    # for t in transform_names:
    #     try:
    #         gd.append(known_list.index(t))
    #     except:
    #         raise KeyError
    
    target = torch.zeros(len(known_list))
    for t in transform_names:
        try:
            target[known_list.index(t)] = 1
        except:
            raise KeyError
    return target