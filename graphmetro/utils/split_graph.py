import torch
import math
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.data import Data
from torch_geometric.utils import degree


def split_batch(data):
    split = degree(data.batch[data.edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(data.edge_index, split, dim=1)
    if not data.edge_attr is None:
        edge_attrs = torch.split(data.edge_attr, split, dim=1)
    else:
        edge_attrs = [None for _ in range(len(edge_indices))]
    num_nodes = degree(data.batch, dtype=torch.long)
    cum_nodes = torch.cat([data.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
    num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(data.x.device)
    cum_edges = torch.cat([data.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])
    data_list = []
    for edge_index, edge_attr, N, C in zip(edge_indices, edge_attrs,
                                           num_nodes, cum_nodes
                                           ):
        data_list.append(Data(x=data.x[C:C+N], 
                              edge_index=edge_index - C, 
                              edge_attr=edge_attr,
                              batch=torch.zeros(N).long()
                              ).to(data.x.device))
    return data_list


# Bidirectional k-hop subgraph
# adapted from torch-geometric.utils.subgraph
def bid_k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                        num_nodes=None):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        if len(subsets) > 1:
            node_mask[subsets[-2]] = True
        edge_mask1 = torch.index_select(node_mask, 0, row)
        edge_mask2 = torch.index_select(node_mask, 0, col)
        subsets.append(col[edge_mask1])
        subsets.append(row[edge_mask2])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask
