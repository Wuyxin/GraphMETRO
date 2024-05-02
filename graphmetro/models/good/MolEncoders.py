"""
Atom (node) and bond (edge) feature encoding specified for molecule data.
"""
import torch
from torch import Tensor


x_map = {
    'atomic_num':
        list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
        list(range(0, 11)),
    'formal_charge':
        list(range(-5, 7)),
    'num_hs':
        list(range(0, 9)),
    'num_radical_electrons':
        list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}


class AtomEncoder(torch.nn.Module):
    r"""
    atom (node) feature encoding specified for molecule data.

    Args:
        emb_dim: number of dimensions of embedding
    """

    def __init__(self, emb_dim):

        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        feat_dims = list(map(len, x_map.values()))

        for i, dim in enumerate(feat_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        r"""
        atom (node) feature encoding specified for molecule data.

        Args:
            x (Tensor): node features

        Returns (Tensor):
            atom (node) embeddings
        """
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    r"""
    bond (edge) feature encoding specified for molecule data.

    Args:
        emb_dim: number of dimensions of embedding
    """

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        edge_feat_dims = list(map(len, e_map.values()))

        for i, dim in enumerate(edge_feat_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        r"""
        bond (edge) feature encoding specified for molecule data.

        Args:
            edge_attr (Tensor): edge attributes

        Returns (Tensor):
            bond (edge) embeddings

        """
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding