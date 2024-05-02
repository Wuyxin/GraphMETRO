r"""
The Graph Neural Network from the `"How Powerful are Graph Neural Networks?"
<https://arxiv.org/abs/1810.00826>`_ paper.
"""
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_sparse import SparseTensor

from graphmetro.models.good.BaseGNN import BasicEncoder
from graphmetro.models.good.MolEncoders import AtomEncoder, BondEncoder
from graphmetro.models.good.Classifiers import Classifier
from torch.nn import Identity
from graphmetro.config import cfg



class GIN(torch.nn.Module):
    r"""
    The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`dim_inner`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.dataset_type`)
    """

    def __init__(self, dim_inner, dim_out, 
                 dim_in=-1, has_glob_pool=False, **kwargs):

        super().__init__()
        self.encoder = GINFeatExtractor(dim_inner)
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
        out = self.classifier(out_readout, edge_index, edge_attr, batch)
        return out


class GINFeatExtractor(torch.nn.Module):
    r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`dim_inner`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.dataset_type`)
    """
    def __init__(self, dim_inner):
        super(GINFeatExtractor, self).__init__()
        self.encoder = GINEncoder(dim_inner)
        self.edge_feat = False

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.torch.nn.Module.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.torch.nn.Module.arguments_read>`

        Returns (Tensor):
            node feature representations
        """
        
        out_readout = self.encoder(x, edge_index, edge_attr, batch)
        return out_readout


class GINEncoder(BasicEncoder):
    r"""
    The GIN encoder for non-molecule data, using the :class:`~GINConv` operator for message passing.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`dim_inner`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`)
    """

    def __init__(self, dim_inner):
        super(GINEncoder, self).__init__()
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

    def forward(self, x, edge_index, edge_weight, batch):
        r"""
        The GIN encoder for non-molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            batch (Tensor): batch indicator
            batch_size (int): batch size

        Returns (Tensor):
            node feature representations
        """

        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        out_readout = self.readout(post_conv, batch)
        return out_readout


class GINMolEncoder(BasicEncoder):
    r"""The GIN encoder for molecule data, using the :class:`~GINEConv` operator for message passing.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`dim_inner`, :obj:`config.model.model_layer`)
    """

    def __init__(self, dim_inner):
        super(GINMolEncoder, self).__init__()
        num_layer = cfg.gnn.layers_mp
        self.atom_encoder = AtomEncoder(dim_inner)

        self.conv1 = GINEConv(nn.Sequential(nn.Linear(dim_inner, 2 * dim_inner),
                                            nn.BatchNorm1d(2 * dim_inner), nn.ReLU(),
                                            nn.Linear(2 * dim_inner, dim_inner)))

        self.convs = nn.ModuleList(
            [
                GINEConv(nn.Sequential(nn.Linear(dim_inner, 2 * dim_inner),
                                       nn.BatchNorm1d(2 * dim_inner), nn.ReLU(),
                                       nn.Linear(2 * dim_inner, dim_inner)))
                for _ in range(num_layer - 1)
            ]
        )

    def forward(self, x, edge_index, edge_attr, batch, batch_size, **kwargs):
        r"""
        The GIN encoder for molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_attr (Tensor): edge attributes
            batch (Tensor): batch indicator
            batch_size (int): Batch size.

        Returns (Tensor):
            node feature representations
        """
        x = self.atom_encoder(x)
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index, edge_attr))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index, edge_attr))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout
    

class GINEConv(gnn.MessagePassing):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 edge_dim: Optional[int] = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        if hasattr(self.nn[0], 'in_features'):
            in_channels = self.nn[0].in_features
        else:
            in_channels = self.nn[0].in_channels
        self.bone_encoder = BondEncoder(in_channels)
        self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x, 
                edge_index: Adj,
                edge_attr: OptTensor = None, 
                size: Size = None
                ) -> Tensor:
        """"""
        if self.bone_encoder:
            edge_attr = self.bone_encoder(edge_attr)
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

