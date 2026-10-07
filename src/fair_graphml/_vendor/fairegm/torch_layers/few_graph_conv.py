from typing import Optional, Tuple

import torch
from torch import nn

from .utils import apply_constraint, apply_initializer


class FEWGraphConv(nn.Module):
    """Fair Edge Weighting layer implemented in PyTorch."""

    def __init__(
        self,
        num_nodes: int,
        kernel_initializer: Optional[str] = None,
        kernel_constraint: Optional[str] = None,
    ):
        super().__init__()
        self.kernel_constraint = kernel_constraint
        self.kernel = nn.Parameter(torch.empty(num_nodes, num_nodes))
        apply_initializer(self.kernel, kernel_initializer or 'glorot_uniform')

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if nodes.dim() == 2:
            nodes = nodes.unsqueeze(0)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        fair_adj = adj * self.kernel
        conv = torch.matmul(fair_adj, nodes)
        return conv, fair_adj

    def apply_constraints(self) -> None:
        apply_constraint(self.kernel, self.kernel_constraint)
