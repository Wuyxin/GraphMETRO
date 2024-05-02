import torch.nn as nn


class Compose(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, x, edge_index, edge_attr, batch):
        _info = {}
        for aug in self.transforms:
            x, edge_index, edge_attr, batch, info = aug(x, edge_index, edge_attr, batch)
            if not info is None:
                for key, item in info.items():
                    if 'node_mask' in _info.keys() and key == 'node_mask':
                        _info[key][_info[key] == 1] = item
                    else:
                        _info[key] = item
        return x, edge_index, edge_attr, batch, _info