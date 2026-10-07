from typing import Optional, Tuple

import torch
from torch import nn

from .utils import apply_constraint, apply_initializer


class GFOGraphConv(nn.Module):
    """Group Fairness Optimisation layer implemented in PyTorch.

    The layer mirrors :class:`layers.gfo_graph_conv.GFOGraphConv` by learning an
    additive correction that is broadcast to every node after the graph
    convolution step.
    """

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        addition_initializer: Optional[str] = None,
        addition_constraint: Optional[str] = None,
    ):
        super().__init__()
        self.addition_constraint = addition_constraint
        self.addition = nn.Parameter(torch.empty(num_nodes, num_features))
        apply_initializer(self.addition, addition_initializer or 'glorot_uniform')

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if nodes.dim() == 2:
            nodes = nodes.unsqueeze(0)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        conv = torch.matmul(adj, nodes)
        fair = conv + self.addition
        return fair, adj

    def apply_constraints(self) -> None:
        apply_constraint(self.addition, self.addition_constraint)
