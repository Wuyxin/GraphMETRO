import torch
import os
import os.path as osp
from torch_geometric.datasets import *
import torch_geometric.transforms as T
from torch_geometric.utils.degree import degree
from tgb.nodeproppred.dataset_pyg import PyGNodePropertyDataset
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset

from graphmetro.utils.seed import set_seed
from graphmetro.datasets.utils import InductiveJointDataset, inductive_rand_split
from graphmetro.config import cfg
from graphmetro.datasets.good import *


def get_datasets(name, root="/dfs/user/shirwu/data"):
    
    if name[:4] == 'GOOD':
        class_name = eval(f'GOOD{name[4:]}')
        datasets, meta_info = class_name.load(dataset_root=root,
                                              shift=cfg.dataset.shift_type)
        
        if meta_info.model_level == 'node':
            dataset = datasets
            datasets = {
                'train': dataset,
                'val': dataset,
                'test': dataset
                }
            for key, dataset in datasets.items():
                setattr(dataset.data, 'y', dataset.data.y.view(-1).long())
                setattr(dataset, 'n_classes', len(torch.unique(dataset.data.y)))
                
        else:
            
            for key, dataset in datasets.items():
                if key in ['task', 'metric']:
                    continue
                if datasets['task'] == 'Binary classification':
                    setattr(dataset, 'n_classes', dataset.data.y.size(-1))
                elif datasets['task'] == 'Regression':
                    setattr(dataset, 'n_classes', 1)
                else:
                    setattr(dataset, 'n_classes', len(torch.unique(dataset.data.y)))
                setattr(dataset.data, 'y', dataset.data.y.view(-1).long())
        return datasets
    
    elif name in ['DBLP', 'CiteSeer', 'PubMed', 'Cora', 
                  'Flickr', 'Reddit', 'GitHub'
                  ]:
        size = {
                'DBLP': 17716,
                'CiteSeer': 3327, 
                'Cora': 2708, 
                'PubMed': 19717, 
                'GitHub': 37700,
                'Reddit': 232965,
                'Flickr': 89250,
                }
        set_seed(cfg.seed)
        pre_transform = T.Compose([
            T.RandomNodeSplit(
                num_val=int(size[name] * cfg.dataset.split[1]), 
                num_test=int(size[name] * cfg.dataset.split[-1])),
            T.TargetIndegree(),
        ])
        
        if name == 'DBLP':
            dataset = CitationFull(osp.join(root, name), 
                                   name=name,
                                   pre_transform=pre_transform)
        elif name == 'Reddit':
            dataset = Reddit2(root=osp.join(root, name))
        elif name == 'Flickr':
            dataset = Flickr(root=osp.join(root, name))
        elif name == 'GitHub':
            dataset = GitHub(osp.join(root, name), 
                             pre_transform=pre_transform)
        else:
            dataset = Planetoid(osp.join(root, name), 
                                name=name,
                                pre_transform=pre_transform)
        datasets = {
            'train': dataset,
            'val': dataset,
            'test': dataset
            }
        for key, dataset in datasets.items():
            setattr(dataset, 'n_classes', len(torch.unique(dataset.data.y)))
        return datasets
    
    elif name[:3] == 'TU_':
        def degree_as_tag(data, max_degree=100):
            d = degree(data.edge_index[0], num_nodes=data.num_nodes) + \
                 degree(data.edge_index[1], num_nodes=data.num_nodes)
            data.x = torch.ones((data.num_nodes, 5)) * d.view(-1, 1) / 100.
            return data
        
        # TU_IMDB doesn't have node features
        if name[3:] in ['IMDB-MULTI', 'REDDIT-BINARY', 'COLLAB']:
            dataset = TUDataset(root, name[3:], transform=degree_as_tag)
        else:
            dataset = TUDataset(root, name[3:])
            
        if not hasattr(dataset, 'splits'):
            datasets = inductive_rand_split(dataset, [0.7, 0.2, 0.1], seed=cfg.seed)
        else:
            datasets = {}
            for key in list(dataset.splits.keys()):
                _id = dataset.splits[key]
                mask = torch.zeros(len(dataset)).bool()
                mask[_id] = True
                datasets[key] = dataset[mask]
        for key, dataset in datasets.items():
            setattr(dataset, 'n_classes', len(torch.unique(dataset.y)))
        return datasets
    
    elif 'ogbn' in name:
        n_classes = 40
        data_root = osp.join(root, 'ogb_dataset')
        os.makedirs(data_root, exist_ok=True)
        dataset = PygNodePropPredDataset(name=name, root=data_root)
        split_idx = dataset.get_idx_split()
        for split in ['train', 'test', 'valid']:
            _split = split[:3] if split == 'valid' else split
            mask = torch.zeros(dataset.data.x.size(0)).bool()
            mask[split_idx[split]] = True
            setattr(dataset.data, f'{_split}_mask', mask)
        setattr(dataset.data, 'y', dataset.data.y.view(-1))
        datasets = {
            'train': dataset,
            'val': dataset,
            'test': dataset
            }
        for key, dataset in datasets.items():
            setattr(dataset, 'n_classes', n_classes)
        return datasets
    
    elif 'ogbg' in name:
        
        n_classes = {'ogbg-ppa': 37}
        dataset = PygGraphPropPredDataset(name=name, transform=T.Constant())
        split_idx = dataset.get_idx_split()
        datasets = {
            'train': dataset[split_idx["train"]],
            'val': dataset[split_idx["valid"]],
            'test': dataset[split_idx["test"]]
            }
        setattr(dataset.data, 'y', dataset.data.y.view(-1))
        for key, dataset in datasets.items():
            setattr(dataset, 'n_classes', n_classes[name])
        return datasets
    
        
def set_dataset_attr(dataset, name, value, size):
    if not hasattr(dataset, 'data') or dataset.data is None:
        setattr(dataset,'data', {})
    dataset._data_list = None
    dataset.data[name] = value
    # error handling --[---]
    if not hasattr(dataset, 'slices') or dataset.slices is None:
        setattr(dataset, 'slices', {})

    dataset.slices[name] = torch.tensor([0, size], dtype=torch.long)