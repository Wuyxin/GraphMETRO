import torch
import copy
import numpy as np


def to_target_index(target, x_dict):
    cnt = 0
    x = torch.concat([x.values()], dim=0)
    for node_type, x in x_dict.items():
        if node_type == target:
            target_index = cnt + torch.arange(len(x.size(0)))
        cnt += x.size(0)
    return target_index
    
    
def dict_to_matrix(x_dict, edge_index_dict, return_x=False):
    all_x = torch.concat(list(x_dict.values()), dim=0)
    if return_x: 
        return all_x
    # collect node types
    x_types = []
    cnt, cum_cnt = 0, {}
    for idx, (node_type, x) in enumerate(x_dict.items()):
        cum_cnt[node_type] = cnt
        cnt += x.size(0)
        x_types.append(torch.ones(x.size(0)).long() * idx)
    x_types = torch.concat(x_types).to(all_x.device)
    # construct edge_index & edge types
    edge_list, edge_types = [], []
    _edge_index_dict = copy.deepcopy(edge_index_dict)
    for idx, (edge_type, edges) in enumerate(_edge_index_dict.items()):
        edges[0, :] = edges[0, :] + cum_cnt[edge_type[0]]
        edges[1, :] = edges[1, :] + cum_cnt[edge_type[-1]]
        edge_list.append(edges)
        edge_types.append(torch.ones(edges.size(-1)).long() * idx)
    edge_index = torch.concat(edge_list, dim=-1)
    edge_types = torch.concat(edge_types).to(all_x.device)
    return all_x, x_types, edge_index, edge_types


def matrix_to_dict(x, x_dict):
    values = [_x for node_type, _x in x_dict.items()]
    n_types = np.array([0] + [x_type_i.size(0) for x_type_i in values])
    cum_n_types = n_types.cumsum()
    keys = list(x_dict.keys())
    for i, key in enumerate(keys):
        x_dict[key] = x[cum_n_types[i] : cum_n_types[i + 1]]
    return x_dict