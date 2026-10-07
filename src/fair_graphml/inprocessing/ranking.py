"""Ranking objectives and the group-conditioned models in the repo."""
import torch
from torch import nn
from torch.nn import functional as F

from ._base import GraphEstimator
from ..data import edge_groups
from ..nn import GraphModel


def log_sinkhorn(log_alpha, n_iter=20):
    """Log-space Sinkhorn normalization, copied from sinkhorn.py."""
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
    return log_alpha.exp()


def ranked_kl_divergence(soft_labels, distribution, top_k=None):
    """Discounted prefix KL(target || prefix), matching the repository loss."""
    k = len(soft_labels) if top_k is None else min(top_k, len(soft_labels))
    if k < 1:
        raise ValueError("top_k must be a positive integer or None.")
    prefix = soft_labels[:k].cumsum(0) / torch.arange(1, k + 1, device=soft_labels.device)[:, None]
    discount = torch.arange(2, k + 2, device=soft_labels.device).float().log2().reciprocal()
    kl = F.kl_div(prefix.clamp_min(1e-10).log(), distribution.expand_as(prefix), reduction="none").sum(1) / soft_labels.shape[1]
    return (discount * kl).sum() / discount.sum()


class Sinkhorn(GraphEstimator):
    """Score-weighted soft-label prefix-KL objective from sinkhorn.py.

    The active fit loop in the source uses softmax(score * group_one_hot),
    rather than its experimental Sinkhorn permutation helper. This class
    preserves that active objective. Extra parameters: sim_coeff=.5,
    group_mode='dyadic', top_k=None.
    """
    requires_sensitive = True
    method_defaults = {"sim_coeff": 0.5, "group_mode": "dyadic", "top_k": None}

    def _setup(self, p):
        super()._setup(p)
        if self.method_params["top_k"] is not None and (not isinstance(self.method_params["top_k"], int) or self.method_params["top_k"] < 1):
            raise ValueError("top_k must be a positive integer or None.")
        if self.task == "link_prediction":
            labels = edge_groups(p.sens, p.edges, self.method_params["group_mode"])
        else:
            labels = p.sens[p.train_mask]
        k = len(p.groups) * (len(p.groups) + 1) // 2 if self.task == "link_prediction" and self.method_params["group_mode"] == "dyadic" else (2 if self.task == "link_prediction" else len(p.groups))
        counts = torch.bincount(labels, minlength=k).float()
        self.target_distribution_ = counts / counts.sum().clamp_min(1)

    def _step(self, p):
        self.optimizer_.zero_grad()
        logits, _ = self._forward(p)
        scores, groups = self._fair_scores(logits, p), self._fair_groups(p)
        one_hot = F.one_hot(groups, len(self.target_distribution_)).float()
        order = scores.argsort(descending=True)
        soft_labels = (scores[:, None] * one_hot).softmax(-1)[order]
        fair = ranked_kl_divergence(soft_labels, self.target_distribution_, self.method_params["top_k"])
        supervised = self._supervised_loss(logits, p)
        loss = supervised + self.method_params["sim_coeff"] * fair
        loss.backward()
        self.optimizer_.step()
        return {"loss": loss.item(), "supervised_loss": supervised.item(), "ranking_loss": fair.item()}


class _GroupModels(nn.Module):
    def __init__(self, models, sens, mode):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.register_buffer("sens", sens)
        self.mode = mode

    def forward(self, x, a, edges=None):
        if len(x) != len(self.sens):
            raise ValueError("ThreeClassifiers is transductive; prediction must retain fitted node IDs.")
        groups = edge_groups(self.sens, edges, self.mode)
        logits = x.new_zeros(edges.shape[1])
        embeddings = []
        for i, model in enumerate(self.models):
            z = model.encoder(x, a)
            selected = groups == i
            if selected.any():
                logits[selected] = model.predictor(z, edges[:, selected])
            embeddings.append(z)
        return logits, torch.stack(embeddings).mean(0)


class ThreeClassifiers(GraphEstimator):
    """Independent encoder/decoder per unordered sensitive group pair.

    For two sensitive groups this gives three classifiers (00, 01, 11), as
    in three_classifiers.py. Extra parameter: group_mode='dyadic'.
    Embeddings are the mean of the independently learned representations.
    """
    requires_sensitive = True
    method_defaults = {"group_mode": "dyadic"}

    def _build(self, p):
        if self.task != "link_prediction":
            raise ValueError("ThreeClassifiers supports link_prediction only.")
        k = len(p.groups) * (len(p.groups) + 1) // 2 if self.method_params["group_mode"] == "dyadic" else 2
        groups = self._fair_groups(p)
        missing = set(range(k)) - set(groups.tolist())
        if missing:
            raise ValueError(f"Training candidates must cover every edge group; missing groups: {sorted(missing)}.")
        models = [GraphModel(self.n_features_in_, self.hidden_channels, self.embedding_dim, 2, self.task, self.dropout, self.encoder, self.decoder) for _ in range(k)]
        return _GroupModels(models, p.sens, self.method_params["group_mode"]).to(self.device)

    def _supervised_loss(self, logits, p):
        groups = self._fair_groups(p)
        return torch.stack([F.binary_cross_entropy_with_logits(logits[groups == g], p.targets[groups == g]) for g in torch.unique(groups)]).mean()


class SinkhornMultiClass(GraphEstimator):
    """Port of the active BCE-only model in three_classifiers_multiclass.py.

    Despite the source class name, its current fit loop trains a single binary
    link predictor and contains no Sinkhorn/ranking loss. This is retained
    explicitly rather than silently substituting a different algorithm.
    """


ThreeClassifiersMultiClass = SinkhornMultiClass
