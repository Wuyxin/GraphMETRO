r"""
The Graph Neural Network from the `"Neural Message Passing for Quantum Chemistry"
<https://proceedings.mlr.press/v70/gilmer17a.html>`_ paper.
"""
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor
from graphmetro.models.good.Classifiers import Classifier
from graphmetro.models.good.Pooling import GlobalAddPool
from graphmetro.models.good.GINs import GINEncoder, GINFeatExtractor, BasicEncoder, GINMolEncoder
from graphmetro.config import cfg


class vGIN(torch.nn.Module):
    r"""
        The Graph Neural Network from the `"Neural Message Passing for Quantum Chemistry"
        <https://proceedings.mlr.press/v70/gilmer17a.html>`_ paper.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`dim_inner`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.dataset_type`, :obj:`cfg.gnn.dropout`)
    """
        
    def __init__(self, dim_inner, dim_out, 
                 dim_in=-1, has_glob_pool=False, 
                 has_post_mp=True, **kwargs):

        super().__init__()
        self.encoder = vGINFeatExtractor(dim_inner)
        self.has_post_mp = has_post_mp
        if self.has_post_mp:
            self.classifier = Classifier(dim_out)
        self.graph_repr = None

    def forward(self, x, edge_index, edge_attr=None, batch=None) -> torch.Tensor:
        r"""
        The GIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.torch.nn.Module.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.torch.nn.Module.arguments_read>`

        Returns (Tensor):
            label predictions

        """
        out_readout = self.encoder(x, edge_index, edge_attr, batch)
        if self.has_post_mp:
            out = self.classifier(out_readout, edge_index, edge_attr, batch)
        return out


class vGINFeatExtractor(GINFeatExtractor):
    r"""
        vGIN feature extractor using the :class:`~vGINEncoder` or :class:`~vGINMolEncoder`.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`dim_inner`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.dataset_type`, :obj:`cfg.gnn.dropout`)
            **kwargs: `without_readout` will output node features instead of graph features.
    """
    def __init__(self, dim_inner):
        super(vGINFeatExtractor, self).__init__(dim_inner)
        if cfg.dataset.name in ['GOODHIV', 'GOODPCBA', 'GOODZINC']:
            self.encoder = vGINMolEncoder(dim_inner)
            self.edge_feat = True
        else:
            self.encoder = vGINEncoder(dim_inner)
            self.edge_feat = False


class vGINEncoder(BasicEncoder):
    r"""
    The vGIN encoder for non-molecule data, using the :class:`~vGINConv` operator for message passing.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`dim_inner`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`cfg.gnn.dropout`)
    """

    def __init__(self, dim_inner):
        super(vGINEncoder, self).__init__()
        num_layer = cfg.gnn.layers_mp

        self.conv1 = gnn.GINConv(nn.Sequential(gnn.Linear(-1, 2 * dim_inner),
                                               nn.BatchNorm1d(2 * dim_inner), 
                                               nn.ReLU(),
                                               nn.Linear(2 * dim_inner, dim_inner)))

        self.convs = nn.ModuleList(
            [
                gnn.GINConv(nn.Sequential(nn.Linear(dim_inner, 2 * dim_inner),
                                      nn.BatchNorm1d(2 * dim_inner), nn.ReLU(),
                                      nn.Linear(2 * dim_inner, dim_inner)))
                for _ in range(num_layer - 1)
            ]
        )
        
        self.virtual_node_embedding = nn.Embedding(1, dim_inner)
        self.virtual_mlp = nn.Sequential(*(
                [nn.Linear(dim_inner, 2 * dim_inner),
                 nn.BatchNorm1d(2 * dim_inner), nn.ReLU()] +
                [nn.Linear(2 * dim_inner, dim_inner),
                 nn.BatchNorm1d(dim_inner), nn.ReLU(),
                 nn.Dropout(cfg.gnn.dropout)]
        ))
        self.virtual_pool = GlobalAddPool()
        
    def forward(self, x, edge_index, edge_weight, batch):
        r"""
        The vGIN encoder for non-molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            batch (Tensor): batch indicator
            batch_size (int): Batch size.

        Returns (Tensor):
            node feature representations
        """
        batch_size = len(torch.unique(batch))
        virtual_node_feat = self.virtual_node_embedding(
            torch.zeros(batch_size, device=cfg.device, dtype=torch.long))

        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            # --- Add global info ---
            post_conv = post_conv + virtual_node_feat[batch]
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
            # --- update global info ---
            if i < len(self.convs) - 1:
                virtual_node_feat = self.virtual_mlp(self.virtual_pool(post_conv, batch) + virtual_node_feat)

        out_readout = self.readout(post_conv, batch)
        return out_readout



class vGINMolEncoder(GINMolEncoder):
    r"""The vGIN encoder for molecule data, using the :class:`~vGINEConv` operator for message passing.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`dim_inner`, :obj:`config.model.model_layer`, :obj:`config.model.dropout_rate`)
    """

    def __init__(self, dim_inner):
        super(vGINMolEncoder, self).__init__(dim_inner)
        self.virtual_node_embedding = nn.Embedding(1, dim_inner)
        self.virtual_mlp = nn.Sequential(*(
                [nn.Linear(dim_inner, 2 * dim_inner),
                 nn.BatchNorm1d(2 * dim_inner), nn.ReLU()] +
                [nn.Linear(2 * dim_inner, dim_inner),
                 nn.BatchNorm1d(dim_inner), nn.ReLU(),
                 nn.Dropout(cfg.gnn.dropout)]
        ))
        self.virtual_pool = GlobalAddPool()

    def forward(self, x, edge_index, edge_attr, batch):
        r"""
        The vGIN encoder for molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_attr (Tensor): edge attributes
            batch (Tensor): batch indicator
            batch_size (int): Batch size.

        Returns (Tensor):
            node feature representations
        """
        batch_size = len(torch.unique(batch))
        virtual_node_feat = self.virtual_node_embedding(
            torch.zeros(batch_size, device=cfg.device, dtype=torch.long))

        x = self.atom_encoder(x)
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index, edge_attr))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            # --- Add global info ---
            post_conv = post_conv + virtual_node_feat[batch]
            post_conv = batch_norm(conv(post_conv, edge_index, edge_attr))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
            # --- update global info ---
            if i < len(self.convs) - 1:
                virtual_node_feat = self.virtual_mlp(self.virtual_pool(post_conv, batch, batch_size) + virtual_node_feat)

        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout