"""Alternating variational graph learning and adjacency fairness projection."""
import torch
from torch import nn
from torch.nn import functional as F

from ._base import GraphEstimator
from ..nn import GraphEncoder, Predictor


def project_simplex(rows):
    """Euclidean projection of each row onto the probability simplex."""
    values = rows.sort(dim=1, descending=True).values
    cumulative = values.cumsum(1) - 1
    ranks = torch.arange(1, rows.shape[1] + 1, device=rows.device)
    rho = (values - cumulative / ranks > 0).sum(1).clamp_min(1)
    theta = cumulative.gather(1, (rho - 1)[:, None]) / rho[:, None]
    return (rows - theta).clamp_min(0)


class _VGAE(nn.Module):
    def __init__(self, owner, p):
        super().__init__()
        self.encoder = GraphEncoder(p.x.shape[1], owner.hidden_channels, 2 * owner.embedding_dim, owner.dropout, owner.encoder)
        self.predictor = Predictor(owner.embedding_dim, 2, owner.task, owner.decoder)

    def forward(self, x, a, edges=None):
        mu, log_std = self.encoder(x, a).chunk(2, dim=-1)
        self.mu, self.log_std = mu, log_std.clamp(-10, 5)
        z = mu + torch.randn_like(mu) * self.log_std.exp() if self.training else mu
        return self.predictor(z, edges), z


class FairAdj(GraphEstimator):
    """FairAdj with VGAE and row-simplex projected adjacency updates.

    Extra parameters: T1=1, T2=5, eta=.01, max_dense_nodes=5000.
    Training uses training candidates exclusively (the source used test edges).
    This method learns a transductive dense N-by-N adjacency.
    """
    requires_sensitive = True
    method_defaults = {"T1": 1, "T2": 5, "eta": 0.01, "max_dense_nodes": 5000}

    def _build(self, p):
        if self.task != "link_prediction":
            raise ValueError("FairAdj supports link_prediction only.")
        if len(p.x) > self.method_params["max_dense_nodes"]:
            raise ValueError("FairAdj uses dense adjacency; increase max_dense_nodes explicitly for this graph.")
        for name in ("T1", "T2"):
            if not isinstance(self.method_params[name], int) or self.method_params[name] < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.adjacency_ = self.adjacency_.to_dense()
        return _VGAE(self, p).to(self.device)

    def _step(self, p):
        m = self.method_params
        for _ in range(m["T1"]):
            self.optimizer_.zero_grad()
            logits, _ = self._forward(p)
            supervised = self._supervised_loss(logits, p)
            mu, ls = self.model_.mu, self.model_.log_std
            kl = -0.5 / len(p.x) * (1 + 2 * ls - mu.square() - (2 * ls).exp()).sum(1).mean()
            loss = supervised + kl
            loss.backward()
            self.optimizer_.step()
        # Equalize mean intra/inter scores over all unordered node pairs.
        pairs = torch.triu_indices(len(p.x), len(p.x), 1, device=p.x.device)
        inter = p.sens[pairs[0]] != p.sens[pairs[1]]
        if not inter.any() or inter.all():
            raise ValueError("FairAdj requires both intra- and inter-group node pairs.")
        self.model_.requires_grad_(False)
        for _ in range(m["T2"]):
            a = self.adjacency_.detach().requires_grad_(True)
            _, z = self.model_(p.x, a, p.target_edges)
            scores = (z[pairs[0]] * z[pairs[1]]).sum(-1)
            fair = (scores[inter].mean() - scores[~inter].mean()).square()
            grad, = torch.autograd.grad(fair, a)
            self.adjacency_ = project_simplex(a - m["eta"] * grad).detach()
        self.model_.requires_grad_(True)
        self.learned_adjacency_ = self.adjacency_.detach().cpu().numpy().copy()
        return {"loss": loss.item(), "supervised_loss": supervised.item(), "kl_loss": kl.item(), "adjacency_fairness_loss": fair.item()}

    def _predict_tensors(self, x, a, edges):
        if len(x) != len(self._prepared_.x):
            raise ValueError("FairAdj requires the fitted node IDs and node count.")
        return self.model_(x, self.adjacency_, edges)


FairADJ = FairAdj
