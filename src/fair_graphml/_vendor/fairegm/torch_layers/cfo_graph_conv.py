from typing import Optional, Tuple

import torch
from torch import nn

from .utils import apply_constraint, apply_initializer


class CFOGraphConv(nn.Module):
    """Community Fairness Optimisation layer implemented in PyTorch."""

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        num_additions: int,
        addition_initializer: Optional[str] = None,
        addition_constraint: Optional[str] = None,
        connection_initializer: Optional[str] = None,
        connection_constraint: Optional[str] = None,
    ):
        super().__init__()
        self.addition_constraint = addition_constraint
        self.connection_constraint = connection_constraint

        self.addition = nn.Parameter(torch.empty(num_additions, num_features))
        self.connection = nn.Parameter(torch.empty(num_nodes, num_additions))

        apply_initializer(self.addition, addition_initializer or 'glorot_uniform')
        apply_initializer(self.connection, connection_initializer or 'glorot_uniform')

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if nodes.dim() == 2:
            nodes = nodes.unsqueeze(0)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        conv = torch.matmul(adj, nodes)
        fair_conv = torch.matmul(self.connection, self.addition)
        fair = conv + fair_conv
        return fair, adj

    def apply_constraints(self) -> None:
        apply_constraint(self.addition, self.addition_constraint)
        apply_constraint(self.connection, self.connection_constraint)
