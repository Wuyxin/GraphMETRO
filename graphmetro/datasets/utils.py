import numpy as np
from typing import Optional, Union
from collections.abc import Sequence

import torch
from torch import Tensor
from torch_geometric.data import InMemoryDataset

from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.data.dataset import Dataset, IndexType
from torch_geometric.data.separate import separate

IndexType = Union[slice, Tensor, np.ndarray, Sequence]


class InductiveJointDataset(InMemoryDataset):
    def __init__(self, datasets: list, transform=None, pre_transform=None):

        self.data = None
        self.transform = transform
        self.pre_transform = pre_transform
        self._indices: Optional[Sequence] = None

        max_dim = 0
        for dataset in datasets:
            if dataset[0].x.size(-1) > max_dim:
                max_dim = dataset[0].x.size(-1)

        data_list = []
        for dataset in datasets:
            for i in range(len(dataset)):
                data = dataset[i]
                if data.x.size(-1) < max_dim:
                    data.x = torch.cat(
                        [data.x, data.x.new_zeros(data.x.size(0), max_dim - data.x.size(-1))], 
                        dim=-1)
                data_list.append(data)

        self.data, self.slices = self.collate(data_list)

        train_mask = torch.zeros(len(data_list), dtype=torch.bool)
        val_mask = torch.zeros(len(data_list), dtype=torch.bool)
        test_mask = torch.zeros(len(data_list), dtype=torch.bool)

        indices = torch.arange(len(data_list))
        train_mask[:len(datasets[0])] = True
        val_mask[len(datasets[0]):len(datasets[0]) + len(datasets[1])] = True
        test_mask[-len(datasets[2]):] = True

        dataset.splits = {
            'train': indices[train_mask], 
            'val': indices[val_mask], 
            'test': indices[test_mask]
            }
            

def inductive_rand_split(dataset, split, seed):

    np.random.seed(seed)
    length = len(dataset)
    indices = torch.from_numpy(np.random.permutation(length)).long()
    cum_indices = [int(sum(split[0:x:1]) * length) for x in range(1, len(split) + 1)] 

    datasets = {}
    train_mask = torch.zeros(len(dataset)).bool()
    test_mask = torch.zeros(len(dataset)).bool()
    val_mask = torch.zeros(len(dataset)).bool() 
    train_mask[indices[:cum_indices[0]]] = True
    test_mask[indices[cum_indices[0]:cum_indices[1]]] = True
    val_mask[indices[cum_indices[1]:]] = True
    for split, mask in zip(['train', 'val', 'test'],
                           [train_mask, val_mask, test_mask]):
        datasets[split] = dataset[mask]
        
    return datasets


from typing import List, Optional, Union
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class Constant(BaseTransform):
    r"""Appends a constant value to each node feature :obj:`x`
    (functional name: :obj:`constant`).

    Args:
        value (float, optional): The value to add. (default: :obj:`1.0`)
        cat (bool, optional): If set to :obj:`False`, existing node features
            will be replaced. (default: :obj:`True`)
        node_types (str or List[str], optional): The specified node type(s) to
            append constant values for if used on heterogeneous graphs.
            If set to :obj:`None`, constants will be added to each node feature
            :obj:`x` for all existing node types. (default: :obj:`None`)
    """
    def __init__(
        self,
        value: float = 1.0,
        cat: bool = True,
        node_types: Optional[Union[str, List[str]]] = None,
    ):
        if isinstance(node_types, str):
            node_types = [node_types]

        self.value = value
        self.cat = cat
        self.node_types = node_types

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

        for store in data.node_stores:
            if self.node_types is None or store._key in self.node_types:
                num_nodes = store.num_nodes
                c = torch.full((num_nodes, 1), self.value, dtype=torch.float)

                if hasattr(store, 'x') and self.cat:
                    x = store.x.view(-1, 1) if store.x.dim() == 1 else store.x
                    store.x = torch.cat([x, c.to(x.device, x.dtype)], dim=-1)
                else:
                    store.x = c

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(value={self.value})'