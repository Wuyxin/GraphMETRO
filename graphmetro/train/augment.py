import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj


def augment(data, augment_type, p):
    if augment_type == 'none':
        return data.x_dict, data.edge_index_dict
    if augment_type == "dropedge":
        edge_index_dict = {}
        for edge_type, edge_index in data.edge_index_dict.items():
            edge_index_dict[edge_type], _ = dropout_adj(edge_index, p=p)
        x_dict = data.x_dict
    else:
        x_dict = {}
        if augment_type == 'shuffle':
            for node_type, x in data.x_dict.items():
                x_dict[node_type] = x[torch.randperm(x.size(0)), :]
        if augment_type == "dropfeat":
            for node_type, x in data.x_dict.items():
                x_dict[node_type] = F.dropout(x, p=p)
        if augment_type == 'noisyfeat':
            for node_type, x in data.x_dict.items():
                x_dict[node_type] = x + torch.randn(x.size()).to(x.device) * p
        edge_index_dict = data.edge_index_dict
    return x_dict, edge_index_dict