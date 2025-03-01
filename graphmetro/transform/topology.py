import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj, add_random_edge, \
                                  is_undirected, dropout_node, dropout_path, k_hop_subgraph, degree

from graphmetro.utils.split_graph import bid_k_hop_subgraph
import warnings


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return x, edge_index, edge_attr, batch, None


class AddEdge(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        force_undirected = is_undirected(edge_index)
        
        num_nodes = degree(batch, dtype=torch.long)
        split = degree(batch[edge_index[0]], dtype=torch.long).tolist()
        cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
        edge_indices = torch.split(edge_index, split, dim=1)
        
        new_edge_index = []
        for edge_index, N, C in zip(edge_indices, num_nodes, cum_nodes):
            _edge_index, _ = add_random_edge(edge_index - C, 
                                             num_nodes=N,
                                             p=self.p,
                                             force_undirected=force_undirected)
            new_edge_index.append(_edge_index + C)
        new_edge_index = torch.concat(new_edge_index, dim=-1)
        
        if not edge_attr is None:
            if len(torch.unique(edge_attr)) == 1:
                edge_attr = torch.ones((edge_index.size(1), edge_attr.size(-1))) * torch.unique(edge_attr).item()
                edge_attr = edge_attr.to(edge_index.device)
            else:
                warnings.warn("Disgard edge attributes if not uniform")
                edge_attr = None
        return x, new_edge_index, edge_attr, batch, None
    
    
class DropEdge(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.p)
        return x, edge_index, edge_attr, batch, None


class DropNode(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        num_nodes = degree(batch, dtype=torch.long)
        split = degree(batch[edge_index[0]], dtype=torch.long, num_nodes=len(num_nodes)).tolist()
        cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
        edge_indices = torch.split(edge_index, split, dim=1)
        assert len(edge_indices) == len(num_nodes)
        cnt = 0
        new_edge_index, edge_masks, node_masks = [], [], []
        for edge_index_i, N, C in zip(edge_indices, num_nodes, cum_nodes):
            while True:
                _edge_index, edge_mask, node_mask = dropout_node(edge_index_i - C, num_nodes=N, p=self.p)
                # avoid none graph
                if torch.sum(node_mask) > 0:
                    break 
            
            node_mapping = -1 * torch.ones(N).long()
            node_mapping[node_mask] = torch.arange(int(torch.sum(node_mask).item()))
            node_mapping = node_mapping.to(x.device)
            row, col = _edge_index
            _edge_index = torch.concat([node_mapping[row].view(1, -1),
                                        node_mapping[col].view(1, -1)], dim=0).long()

            new_edge_index.append(_edge_index + cnt)
            edge_masks.append(edge_mask)
            node_masks.append(node_mask)
            cnt += int(torch.sum(node_masks[-1]))
            assert edge_mask.sum() == _edge_index.size(-1)
            
        new_edge_index = torch.concat(new_edge_index, dim=-1)
        edge_mask = torch.concat(edge_masks, dim=0)
        node_mask = torch.concat(node_masks, dim=0)
        
        if not edge_attr is None:
            edge_attr = edge_attr[edge_mask]
        if not batch is None:
            batch = batch[node_mask]
                
                
        return x[node_mask], new_edge_index, edge_attr, batch, {'node_mask': node_mask}
            


class DropPath(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        edge_index, edge_mask = dropout_path(edge_index, p=self.p)
        if not edge_attr is None:
            edge_attr = edge_attr[edge_mask]
        return x, edge_index, edge_attr, batch, {'edge_mask': edge_mask}


class RandomSubgraph(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = int(k)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        node_idx = []
        if not batch is None:
            for b in torch.unique(batch):
                node_idx.append(torch.randint(low=int(torch.sum(batch < b)), 
                                              high=int(torch.sum(batch <= b)), 
                                              size=(1,)).item())
        subset, edge_index, mapping, edge_mask = bid_k_hop_subgraph(node_idx, 
                                                                    num_hops=self.k, 
                                                                    edge_index=edge_index, 
                                                                    num_nodes=x.size(0),
                                                                    relabel_nodes=True
                                                                    )
        if not edge_attr is None:
            edge_attr = edge_attr[edge_mask]
        if not batch is None:
            batch = batch[subset]
        node_mask = torch.zeros(x.size(0)).bool().to(x.device)
        node_mask[subset] = True
        return x[subset], edge_index, edge_attr, batch, {'node_mask': node_mask}
            
    