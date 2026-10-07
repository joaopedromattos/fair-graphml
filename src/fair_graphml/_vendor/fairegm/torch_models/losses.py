from typing import Callable

import torch
from torch import Tensor
from torch.nn import functional as F


def _kl_divergence(p: Tensor, q: Tensor) -> Tensor:
    q = q.clamp_min(1e-7)
    p = p.clamp_min(1e-7)
    return torch.sum(p * torch.log(p / q), dim=-1)


def dp_link_divergence_loss(attributes: Tensor, edges: Tensor) -> Tensor:
    edges = torch.sigmoid(edges)
    f = torch.matmul(edges, attributes) + 1e-7
    e = torch.sum(attributes, dim=1) + 1e-7
    norme = torch.sum(e, dim=-1, keepdim=True)
    normf = torch.sum(f, dim=-1, keepdim=True)
    true = e / norme
    pred = f / normf
    return _kl_divergence(true, pred).mean()


def build_reconstruction_loss(pos_weight: float) -> Callable[[Tensor, Tensor], Tensor]:
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)

    def reconstruction_loss(true_adj: Tensor, pred_adj: Tensor) -> Tensor:
        true_adj = true_adj.to(pred_adj.dtype)
        return F.binary_cross_entropy_with_logits(
            pred_adj, true_adj, pos_weight=pos_weight_tensor.to(pred_adj.device), reduction='mean'
        )

    return reconstruction_loss


def build_dp_link_divergence_loss_vgae(embedding_dim: int) -> Callable[[Tensor, Tensor], Tensor]:
    def dp_loss(attributes: Tensor, pred: Tensor) -> Tensor:
        edges = torch.sigmoid(pred[..., :-2 * embedding_dim])
        f = torch.matmul(edges, attributes) + 1e-7
        e = torch.sum(attributes, dim=1) + 1e-7
        norme = torch.sum(e, dim=-1, keepdim=True)
        normf = torch.sum(f, dim=-1, keepdim=True)
        true = e / norme
        pred_norm = f / normf
        return _kl_divergence(true, pred_norm).mean()

    return dp_loss


def build_reconstruction_loss_vgae(
    pos_weight: float,
    norm: float,
    num_nodes: int,
    embedding_dim: int,
) -> Callable[[Tensor, Tensor], Tensor]:
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)

    def reconstruction_loss(true_adj: Tensor, pred: Tensor) -> Tensor:
        true_adj = true_adj.to(pred.dtype)
        pred_adj = pred[..., :num_nodes]
        z_mean = pred[..., num_nodes:-embedding_dim]
        z_log_std = pred[..., -embedding_dim:]

        entropy = norm * F.binary_cross_entropy_with_logits(
            pred_adj, true_adj, pos_weight=pos_weight_tensor.to(pred.device), reduction='mean'
        )
        kl = (0.5 / num_nodes) + torch.sum(
            1 + 2 * z_log_std - z_mean.pow(2) - torch.exp(2 * z_log_std), dim=-1
        ).mean()
        return entropy - kl

    return reconstruction_loss
