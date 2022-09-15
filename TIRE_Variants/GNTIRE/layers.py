from __future__ import division

import numbers
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import Parameter

import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_geometric.nn.inits import glorot, zeros

# Sourced originally from PyTorch Geometric Temporal @ https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/nn/attention/mtgnn.html#GraphConstructor

class Linear(nn.Module):
    r"""An implementation of the linear layer, conducting 2D convolution.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        bias (bool, optional): Whether to have bias. Default: True.
    """

    def __init__(self, c_in: int, c_out: int, bias: bool = True):
        super(Linear, self).__init__()
        self._mlp = torch.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the linear layer.

        Arg types:
            * **X** (Pytorch Float Tensor) - Input tensor, with shape (batch_size, c_in, num_nodes, seq_len).

        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, c_out, num_nodes, seq_len).
        """
        return self._mlp(X)


class MixProp(nn.Module):
    r"""An implementation of the dynatic mix-hop propagation layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        gdep (int): Depth of graph convolution.
        dropout (float): Dropout rate.
        alpha (float): Ratio of retaining the root nodes's original states, a value between 0 and 1.
    """

    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float):
        super(MixProp, self).__init__()
        self._mlp = Linear((gdep + 1) * c_in, c_out)
        self._gdep = gdep
        self._dropout = dropout
        self._alpha = alpha

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor, A: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of mix-hop propagation.

        Arg types:
            * **X** (Pytorch Float Tensor) - Input feature Tensor, with shape (batch_size, c_in, num_nodes, seq_len).
            * **A** (PyTorch Float Tensor) - Adjacency matrix, with shape (num_nodes, num_nodes).

        Return types:
            * **H_0** (PyTorch Float Tensor) - Hidden representation for all nodes, with shape (batch_size, c_out, num_nodes, seq_len).
        """
        A = A + torch.eye(A.size(0)).to(X.device)
        d = A.sum(1)
        H = X
        H_0 = X
        A = A / d.view(-1, 1)
        for _ in range(self._gdep):
            H = self._alpha * X + (1 - self._alpha) * torch.einsum(
                "ncwl,vw->ncvl", (H, A)
            )
            H_0 = torch.cat((H_0, H), dim=1)
        H_0 = self._mlp(H_0)
        return H_0


class DilatedInception(nn.Module):
    r"""An implementation of the dilated inception layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        kernel_set (list of int): List of kernel sizes.
        dilated_factor (int, optional): Dilation factor.
    """

    def __init__(self, c_in: int, c_out: int, kernel_set: list, dilation_factor: int):
        super(DilatedInception, self).__init__()
        self._time_conv = nn.ModuleList()
        self._kernel_set = kernel_set
        c_out = int(c_out / len(self._kernel_set))
        for kern in self._kernel_set:
            self._time_conv.append(
                nn.Conv2d(c_in, c_out, (1, kern), dilation=(1, dilation_factor))
            )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X_in: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of dilated inception.

        Arg types:
            * **X_in** (Pytorch Float Tensor) - Input feature Tensor, with shape (batch_size, c_in, num_nodes, seq_len).

        Return types:
            * **X** (PyTorch Float Tensor) - Hidden representation for all nodes,
            with shape (batch_size, c_out, num_nodes, seq_len-6).
        """
        X = []
        for i in range(len(self._kernel_set)):
            X.append(self._time_conv[i](X_in))
        for i in range(len(self._kernel_set)):
            X[i] = X[i][..., -X[-1].size(3) :]
        X = torch.cat(X, dim=1)
        return X


class GraphConstructor(nn.Module):
    r"""An implementation of the graph learning layer to construct an adjacency matrix.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        nnodes (int): Number of nodes in the graph.
        k (int): Number of largest values to consider in constructing the neighbourhood of a node (pick the "nearest" k nodes).
        dim (int): Dimension of the node embedding.
        alpha (float, optional): Tanh alpha for generating adjacency matrix, alpha controls the saturation rate
        xd (int, optional): Static feature dimension, default None.
    """

    def __init__(
        self, nnodes: int, k: int, dim: int, alpha: float, xd: Optional[int] = None
    ):
        super(GraphConstructor, self).__init__()
        if xd is not None:
            self._static_feature_dim = xd
            self._linear1 = nn.Linear(xd, dim)
            self._linear2 = nn.Linear(xd, dim)
        else:
            self._embedding1 = nn.Embedding(nnodes, dim)
            self._embedding2 = nn.Embedding(nnodes, dim)
            self._linear1 = nn.Linear(dim, dim)
            self._linear2 = nn.Linear(dim, dim)

        self._k = k
        self._alpha = alpha

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self, idx: torch.LongTensor, FE: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """
        Making a forward pass to construct an adjacency matrix from node embeddings.

        Arg types:
            * **idx** (Pytorch Long Tensor) - Input indices, a permutation of the number of nodes, default None (no permutation).
            * **FE** (Pytorch Float Tensor, optional) - Static feature, default None.
        Return types:
            * **A** (PyTorch Float Tensor) - Adjacency matrix constructed from node embeddings.
        """

        if FE is None:
            nodevec1 = self._embedding1(idx)
            nodevec2 = self._embedding2(idx)
        else:
            assert FE.shape[1] == self._static_feature_dim
            nodevec1 = FE[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self._alpha * self._linear1(nodevec1))
        nodevec2 = torch.tanh(self._alpha * self._linear2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(
            nodevec2, nodevec1.transpose(1, 0)
        )
        A = F.relu(torch.tanh(self._alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(A.device)
        mask.fill_(float("0"))
        s1, t1 = A.topk(self._k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        A = A * mask
        return A

class MTGNNLayer(nn.Module):
    r"""An implementation of the MTGNN layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        dilation_exponential (int): Dilation exponential.
        rf_size_i (int): Size of receptive field.
        kernel_size (int): Size of kernel for convolution, to calculate receptive field size.
        j (int): Iteration index.
        residual_channels (int): Residual channels.
        conv_channels (int): Convolution channels.
        skip_channels (int): Skip channels.
        kernel_set (list of int): List of kernel sizes.
        new_dilation (int): Dilation.
        layer_norm_affline (bool): Whether to do elementwise affine in Layer Normalization.
        gcn_true (bool): Whether to add graph convolution layer.
        seq_length (int): Length of input sequence.
        receptive_field (int): Receptive field.
        dropout (float): Droupout rate.
        gcn_depth (int): Graph convolution depth.
        num_nodes (int): Number of nodes in the graph.
        propalpha (float): Prop alpha, ratio of retaining the root nodes's original states in mix-hop propagation, a value between 0 and 1.

    """

    def __init__(
        self,
        dilation_exponential: int,
        rf_size_i: int,
        kernel_size: int,
        j: int,
        residual_channels: int,
        conv_channels: int,
        skip_channels: int,
        kernel_set: list,
        new_dilation: int,
        layer_norm_affline: bool,
        gcn_true: bool,
        seq_length: int,
        receptive_field: int,
        dropout: float,
        gcn_depth: int,
        num_nodes: int,
        propalpha: float,
    ):
        super(MTGNNLayer, self).__init__()
        self._dropout = dropout
        self._gcn_true = gcn_true

        if dilation_exponential > 1:
            rf_size_j = int(
                rf_size_i
                + (kernel_size - 1)
                * (dilation_exponential ** j - 1)
                / (dilation_exponential - 1)
            )
        else:
            rf_size_j = rf_size_i + j * (kernel_size - 1)

        self._filter_conv = DilatedInception(
            residual_channels,
            conv_channels,
            kernel_set=kernel_set,
            dilation_factor=new_dilation,
        )

        self._gate_conv = DilatedInception(
            residual_channels,
            conv_channels,
            kernel_set=kernel_set,
            dilation_factor=new_dilation,
        )

        self._residual_conv = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=residual_channels,
            kernel_size=(1, 1),
        )

        if seq_length > receptive_field:
            self._skip_conv = nn.Conv2d(
                in_channels=conv_channels,
                out_channels=skip_channels,
                kernel_size=(1, seq_length - rf_size_j + 1),
            )
        else:
            self._skip_conv = nn.Conv2d(
                in_channels=conv_channels,
                out_channels=skip_channels,
                kernel_size=(1, receptive_field - rf_size_j + 1),
            )

        if gcn_true:
            self._mixprop_conv1 = MixProp(
                conv_channels, residual_channels, gcn_depth, dropout, propalpha
            )

            self._mixprop_conv2 = MixProp(
                conv_channels, residual_channels, gcn_depth, dropout, propalpha
            )

        if seq_length > receptive_field:
            self._normalization = LayerNormalization(
                (residual_channels, num_nodes, seq_length - rf_size_j + 1),
                elementwise_affine=layer_norm_affline,
            )

        else:
            self._normalization = LayerNormalization(
                (residual_channels, num_nodes, receptive_field - rf_size_j + 1),
                elementwise_affine=layer_norm_affline,
            )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self,
        X: torch.FloatTensor,
        X_skip: torch.FloatTensor,
        A_tilde: Optional[torch.FloatTensor],
        idx: torch.LongTensor,
        training: bool,
    ) -> torch.FloatTensor:
        """
        Making a forward pass of MTGNN layer.

        Arg types:
            * **X** (PyTorch FloatTensor) - Input feature tensor,
                with shape (batch_size, in_dim, num_nodes, seq_len).
            * **X_skip** (PyTorch FloatTensor) - Input feature tensor for skip connection,
                with shape (batch_size, in_dim, num_nodes, seq_len).
            * **A_tilde** (Pytorch FloatTensor or None) - Predefined adjacency matrix.
            * **idx** (Pytorch LongTensor) - Input indices.
            * **training** (bool) - Whether in traning mode.

        Return types:
            * **X** (PyTorch FloatTensor) - Output sequence tensor,
                with shape (batch_size, seq_len, num_nodes, seq_len).
            * **X_skip** (PyTorch FloatTensor) - Output feature tensor for skip connection,
                with shape (batch_size, in_dim, num_nodes, seq_len).
        """
        X_residual = X
        X_filter = self._filter_conv(X)
        X_filter = torch.tanh(X_filter)
        X_gate = self._gate_conv(X)
        X_gate = torch.sigmoid(X_gate)
        X = X_filter * X_gate
        X = F.dropout(X, self._dropout, training=training)
        X_skip = self._skip_conv(X) + X_skip
        if self._gcn_true:
            X = self._mixprop_conv1(X, A_tilde) + self._mixprop_conv2(
                X, A_tilde.transpose(1, 0)
            )
        else:
            X = self._residual_conv(X)

        X = X + X_residual[:, :, :, -X.size(3) :]
        X = self._normalization(X, idx)
        return X, X_skip


class GConvGRU(torch.nn.Module):
    r"""An implementation of the Chebyshev Graph Convolutional Gated Recurrent Unit
    Cell. For details see this paper: `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(GConvGRU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):

        self.conv_x_z = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_z = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_x_r = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_r = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_x_h = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_h = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros( torch.Size(list(X.shape)[:-1] + [self.out_channels])).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H, lambda_max):
        Z = self.conv_x_z(X, edge_index, edge_weight, lambda_max=lambda_max)
        Z_add = self.conv_h_z(H, edge_index, edge_weight, lambda_max=lambda_max)

        Z = Z + Z_add
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H, lambda_max):
        R = self.conv_x_r(X, edge_index, edge_weight, lambda_max=lambda_max)
        R = R + self.conv_h_r(H, edge_index, edge_weight, lambda_max=lambda_max)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R, lambda_max):
        H_tilde = self.conv_x_h(X, edge_index, edge_weight, lambda_max=lambda_max)
        H_tilde = H_tilde + self.conv_h_h(H * R, edge_index, edge_weight, lambda_max=lambda_max)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.


        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H, lambda_max)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H, lambda_max)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R, lambda_max)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H

class GConvLSTM(torch.nn.Module):
    r"""An implementation of the Chebyshev Graph Convolutional Long Short Term Memory
    Cell. For details see this paper: `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(GConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):

        self.conv_x_i = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_i = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):

        self.conv_x_f = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_f = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):

        self.conv_x_c = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_c = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):

        self.conv_x_o = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_o = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros( torch.Size(list(X.shape)[:-1] + [self.out_channels])).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros( torch.Size(list(X.shape)[:-1] + [self.out_channels])).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        I = self.conv_x_i(X, edge_index, edge_weight, lambda_max=lambda_max)
        I = I + self.conv_h_i(H, edge_index, edge_weight, lambda_max=lambda_max)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        F = self.conv_x_f(X, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + self.conv_h_f(H, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F, lambda_max):
        T = self.conv_x_c(X, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.conv_h_c(H, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        O = self.conv_x_o(X, edge_index, edge_weight, lambda_max=lambda_max)
        O = O + self.conv_h_o(H, edge_index, edge_weight, lambda_max=lambda_max)
        O = O + (self.w_c_o * C)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H, C, lambda_max)
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C, lambda_max)
        C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F, lambda_max)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H, C, lambda_max)
        H = self._calculate_hidden_state(O, C)
        return H, C
