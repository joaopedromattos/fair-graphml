import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
class DenseGraphConv(nn.Module):
    """Native torch equivalent of PyG DenseGraphConv with additive aggregation."""
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.lin_rel = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_root = nn.Linear(in_channels, out_channels, bias=False)
    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()
    def forward(self, x, adj):
        return self.lin_rel(adj @ x) + self.lin_root(x)



def _get_activation(name: Optional[str]):
    if name is None or name == 'linear':
        return None
    name = name.lower()
    if not hasattr(F, name):
        raise ValueError(f"Unsupported activation '{name}'.")
    return getattr(F, name)


class GraphCNN(nn.Module):
    """Dense graph convolution layer compatible with the TensorFlow implementation.

    This layer mirrors :class:`layers.graph_cnn.GraphCNN` but relies on
    :class:`torch_geometric.nn.DenseGraphConv` for the matrix multiplications so
    that the implementation remains faithful while leveraging PyG operators.
    Inputs are expected to be dense adjacency matrices and dense node feature
    matrices with an optional batch dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Optional[str] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = DenseGraphConv(in_channels, out_channels, bias=bias)
        self.activation = _get_activation(activation)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.conv.reset_parameters()

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the dense graph convolution.

        Args:
            nodes: Tensor of shape ``(batch, num_nodes, in_channels)`` or
                ``(num_nodes, in_channels)``.
            adj: Dense adjacency tensor of shape ``(batch, num_nodes, num_nodes)``
                or ``(num_nodes, num_nodes)``.

        Returns:
            Tuple of the convolved node embeddings and the (unchanged) adjacency
            matrix so that the layer can be chained identically to the TensorFlow
            version.
        """
        if nodes.dim() == 2:
            nodes = nodes.unsqueeze(0)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)

        out = self.conv(nodes, adj)
        if self.activation is not None:
            out = self.activation(out)
        return out, adj
