"""Supervised fairness objectives ported from the repository's experiments."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ._base import GraphEstimator
from ..nn import GraphEncoder, adjacency


class GAE(GraphEstimator):
    """Two-layer graph autoencoder with an inner-product decoder by default."""


class FairGNN(GraphEstimator):
    """Sensitive-label estimation, covariance regularization, and adversarial learning.

    Extra parameters: alpha=4., beta=.01, estimator_weight=1.
    Sensitive supervision can be restricted with sensitive_train_mask in fit.
    """
    requires_sensitive = True
    method_defaults = {"alpha": 4.0, "beta": 0.01, "estimator_weight": 1.0}

    def _setup(self, p):
        k = len(p.groups)
        self.model_.sensitive_estimator = GraphEncoder(self.n_features_in_, self.hidden_channels, k, self.dropout).to(self.device)
        self.model_.adversary = nn.Linear(self.embedding_dim, k).to(self.device)
        self.optimizer_ = torch.optim.Adam(
            [v for name, v in self.model_.named_parameters() if not name.startswith("adversary.")],
            lr=self.lr, weight_decay=self.weight_decay)
        self.adversary_optimizer_ = torch.optim.Adam(self.model_.adversary.parameters(), lr=self.lr)

    def _step(self, p):
        self.model_.adversary.requires_grad_(False)
        self.optimizer_.zero_grad()
        logits, z = self._forward(p)
        est = self.model_.sensitive_estimator(p.x, self.adjacency_)
        estimated = est.softmax(-1).detach().clone()
        estimated[p.sensitive_mask] = F.one_hot(p.sens[p.sensitive_mask], len(p.groups)).float()
        if self.task == "link_prediction":
            s = torch.cat((estimated[p.target_edges[0]], estimated[p.target_edges[1]]), dim=1)
            prediction = logits.sigmoid().unsqueeze(-1)
        else:
            s, prediction = estimated[p.train_mask], logits[p.train_mask].softmax(-1)
        cov = ((s - s.mean(0)).t() @ (prediction - prediction.mean(0)) / len(s)).abs().mean()
        adv = -(estimated * self.model_.adversary(z).log_softmax(-1)).sum(-1).mean()
        estimator = F.cross_entropy(est[p.sensitive_mask], p.sens[p.sensitive_mask])
        supervised = self._supervised_loss(logits, p)
        loss = supervised + self.method_params["alpha"] * cov - self.method_params["beta"] * adv + self.method_params["estimator_weight"] * estimator
        loss.backward()
        self.optimizer_.step()
        self.model_.adversary.requires_grad_(True)
        self.adversary_optimizer_.zero_grad()
        adversarial_loss = -(estimated * self.model_.adversary(z.detach()).log_softmax(-1)).sum(-1).mean()
        adversarial_loss.backward()
        self.adversary_optimizer_.step()
        return {"loss": loss.item(), "supervised_loss": supervised.item(), "covariance": cov.item(), "adversary_loss": adv.item()}


class NIFTY(GraphEstimator):
    """Counterfactual feature/edge augmentation and stop-gradient Siamese learning.

    Extra parameters: sim_coeff=.5, drop_edge_rate=.1, drop_feature_rate=.1,
    sensitive_feature_index=None. Set the index to flip a binary sensitive
    feature, or leave it None when that feature is absent from data.x.
    """
    requires_sensitive = True
    method_defaults = {"sim_coeff": 0.5, "drop_edge_rate": 0.1, "drop_feature_rate": 0.1,
                       "sensitive_feature_index": None}

    def _setup(self, p):
        for key in ("sim_coeff", "drop_edge_rate", "drop_feature_rate"):
            if not 0 <= self.method_params[key] < 1:
                raise ValueError(f"{key} must be in [0, 1).")
        dim = self.embedding_dim
        self.model_.projector = nn.Sequential(nn.utils.spectral_norm(nn.Linear(dim, dim)), nn.BatchNorm1d(dim), nn.ReLU(), nn.utils.spectral_norm(nn.Linear(dim, dim)), nn.BatchNorm1d(dim)).to(self.device)
        self.model_.projection_predictor = nn.Sequential(nn.utils.spectral_norm(nn.Linear(dim, dim)), nn.BatchNorm1d(dim), nn.ReLU(), nn.utils.spectral_norm(nn.Linear(dim, dim))).to(self.device)
        if len(p.x) < 2:
            raise ValueError("NIFTY requires at least two nodes for batch normalization.")
        self.representation_optimizer_ = torch.optim.Adam(
            list(self.model_.encoder.parameters()) + list(self.model_.projector.parameters()) + list(self.model_.projection_predictor.parameters()),
            lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer_ = torch.optim.Adam(list(self.model_.encoder.parameters()) + list(self.model_.predictor.parameters()),
                                           lr=self.lr, weight_decay=self.weight_decay)

    def _view(self, p, flip=False):
        m = self.method_params
        keep = torch.rand(p.edges.shape[1], device=p.x.device) >= m["drop_edge_rate"]
        edges = p.edges[:, keep]
        weights = None if p.weights is None else p.weights[keep]
        a = adjacency(edges, len(p.x), weights, self_loops=True)
        x = p.x.clone()
        columns = torch.rand(x.shape[1], device=x.device) < m["drop_feature_rate"]
        # Match the repository's Gaussian feature perturbation.
        x[:, columns] += torch.randn((), device=x.device)
        index = m["sensitive_feature_index"]
        if index is not None:
            if not isinstance(index, int) or not 0 <= index < x.shape[1]:
                raise ValueError("sensitive_feature_index must index a feature column.")
            if not ((p.x[:, index] == 0) | (p.x[:, index] == 1)).all():
                raise ValueError("Counterfactual flips require a binary sensitive feature column.")
            x[:, index] = 1 - p.x[:, index] if flip else p.x[:, index]
        return x, a

    def _step(self, p):
        self.representation_optimizer_.zero_grad()
        self.optimizer_.zero_grad()
        x1, a1 = self._view(p)
        x2, a2 = self._view(p, flip=True)
        logits1, z1 = self.model_(x1, a1, p.target_edges)
        logits2, z2 = self.model_(x2, a2, p.target_edges)
        q1, q2 = self.model_.projector(z1), self.model_.projector(z2)
        h1, h2 = self.model_.projection_predictor(q1), self.model_.projection_predictor(q2)
        mask = p.train_mask
        sim = -(F.cosine_similarity(h1[mask], q2[mask].detach()).mean() + F.cosine_similarity(h2[mask], q1[mask].detach()).mean()) / 2
        coeff = self.method_params["sim_coeff"]
        (coeff * sim).backward()
        self.representation_optimizer_.step()
        # Recompute encoder outputs after the representation update, as in
        # the source's two-optimizer training schedule.
        self.optimizer_.zero_grad()
        logits1, _ = self.model_(x1, a1, p.target_edges)
        logits2, _ = self.model_(x2, a2, p.target_edges)
        supervised = (self._supervised_loss(logits1, p) + self._supervised_loss(logits2, p)) / 2
        ((1 - coeff) * supervised).backward()
        self.optimizer_.step()
        loss = coeff * sim.detach() + (1 - coeff) * supervised.detach()
        return {"loss": loss.item(), "supervised_loss": supervised.item(), "similarity_loss": sim.item()}


class FairVGNN(GraphEstimator):
    """Learned feature-channel masks, adversarial confusion, and weight clipping.

    Extra parameters: alpha=1., mask_regularization=1., temperature=1.,
    clip=1., adversary_steps=1, generator_steps=1, classifier_steps=1.
    """
    requires_sensitive = True
    method_defaults = {"alpha": 1.0, "mask_regularization": 1.0, "temperature": 1.0,
                       "clip": 1.0, "adversary_steps": 1, "generator_steps": 1,
                       "classifier_steps": 1}

    def _setup(self, p):
        if self.method_params["temperature"] <= 0 or self.method_params["clip"] <= 0:
            raise ValueError("temperature and clip must be positive.")
        for key in ("adversary_steps", "generator_steps", "classifier_steps"):
            if not isinstance(self.method_params[key], int) or self.method_params[key] < 1:
                raise ValueError(f"{key} must be a positive integer.")
        self.model_.mask_logits = nn.Parameter(torch.zeros(self.n_features_in_, 2, device=self.device))
        self.model_.adversary = nn.Linear(self.embedding_dim, len(p.groups)).to(self.device)
        self.optimizer_ = torch.optim.Adam(list(self.model_.encoder.parameters()) + list(self.model_.predictor.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        self.mask_optimizer_ = torch.optim.Adam([self.model_.mask_logits], lr=self.lr)
        self.adversary_optimizer_ = torch.optim.Adam(self.model_.adversary.parameters(), lr=self.lr)

    def _mask(self, training=True):
        if training:
            return F.gumbel_softmax(self.model_.mask_logits, tau=self.method_params["temperature"], hard=False)[:, 0]
        return self.model_.mask_logits.softmax(-1)[:, 0]

    def _forward(self, p):
        return self.model_(p.x * self._mask(False), self.adjacency_, p.target_edges)

    def _predict_tensors(self, x, a, edges):
        return self.model_(x * self._mask(False), a, edges)

    def _step(self, p):
        m = self.method_params
        for _ in range(m["adversary_steps"]):
            self.adversary_optimizer_.zero_grad()
            self.optimizer_.zero_grad()
            z = self.model_.encoder(p.x * self._mask().detach(), self.adjacency_)
            adv = F.cross_entropy(self.model_.adversary(z[p.sensitive_mask]), p.sens[p.sensitive_mask])
            adv.backward()
            self.adversary_optimizer_.step()
            self.optimizer_.step()
        for _ in range(m["classifier_steps"]):
            self.optimizer_.zero_grad()
            logits, _ = self.model_(p.x * self._mask().detach(), self.adjacency_, p.target_edges)
            supervised = self._supervised_loss(logits, p)
            supervised.backward()
            self.optimizer_.step()
        self.model_.adversary.requires_grad_(False)
        for _ in range(m["generator_steps"]):
            self.optimizer_.zero_grad()
            self.mask_optimizer_.zero_grad()
            mask = self._mask()
            z = self.model_.encoder(p.x * mask, self.adjacency_)
            probabilities = self.model_.adversary(z[p.sensitive_mask]).softmax(-1)
            confusion = F.mse_loss(probabilities, torch.full_like(probabilities, 1 / len(p.groups)))
            generator = m["alpha"] * confusion + m["mask_regularization"] * F.mse_loss(mask, torch.ones_like(mask))
            generator.backward()
            self.optimizer_.step()
            self.mask_optimizer_.step()
        self.model_.adversary.requires_grad_(True)
        # The input-channel contribution is bounded using the learned mask.
        with torch.no_grad():
            weights = self._mask(False)
            bound = m["clip"] / weights.clamp_min(1e-4)
            first = self.model_.encoder.lin1.weight
            if self.encoder == "sage":
                bound = bound.repeat(2)
            first.copy_(torch.maximum(torch.minimum(first, bound[None, :]), -bound[None, :]))
        return {"loss": (supervised + generator).item(), "supervised_loss": supervised.item(), "adversary_loss": adv.item(), "generator_loss": generator.item()}


class DELTR(GraphEstimator):
    """Disparate-exposure regularization for graph link or node predictions.

    Extra parameters: exposure_coeff=.5, group_mode='inter',
    protected_group=1. Exposure is a stable softmax probability per group
    member, with the one-sided squared penalty used by DELTR.
    """
    requires_sensitive = True
    method_defaults = {"exposure_coeff": 0.5, "group_mode": "inter", "protected_group": 1}

    def _step(self, p):
        self.optimizer_.zero_grad()
        logits, _ = self._forward(p)
        scores = self._fair_scores(logits, p)
        groups = self._fair_groups(p)
        protected = groups == self.method_params["protected_group"]
        if not protected.any() or protected.all():
            raise ValueError("DELTR training candidates must include protected and unprotected groups.")
        exposure = scores.softmax(0)
        diff = exposure[~protected].mean() - exposure[protected].mean()
        fair = diff.clamp_min(0).square()
        supervised = self._supervised_loss(logits, p)
        loss = supervised + self.method_params["exposure_coeff"] * fair
        loss.backward()
        self.optimizer_.step()
        return {"loss": loss.item(), "supervised_loss": supervised.item(), "exposure_loss": fair.item()}


class UGE(GraphEstimator):
    """Unbiased graph embedding via group-pair importance weighting.

    Extra parameters: debias_method='uge-w' ('uge-r' or 'uge-c'), reg_weight=.5.
    Weighting uses possible group-pair frequencies divided by observed positive
    group-pair frequencies. The regularizer balances predicted group rates.
    """
    requires_sensitive = True
    method_defaults = {"debias_method": "uge-w", "reg_weight": 0.5, "group_mode": "dyadic"}

    def _setup(self, p):
        if self.task != "link_prediction":
            raise ValueError("UGE supports link_prediction; use transform() for downstream node classifiers.")
        if self.method_params["debias_method"] not in {"uge-w", "uge-r", "uge-c"}:
            raise ValueError("debias_method must be 'uge-w', 'uge-r', or 'uge-c'.")
        super()._setup(p)
        groups = self._fair_groups(p)
        positive = p.targets == 1
        counts = torch.bincount(p.sens, minlength=len(p.groups)).float()
        possible = []
        for a in range(len(counts)):
            for b in range(a, len(counts)):
                possible.append(counts[a] * counts[b] if a != b else counts[a] * (counts[a] - 1) / 2)
        possible = torch.stack(possible)
        observed = torch.bincount(groups[positive], minlength=len(possible)).float()
        ratio = possible / possible.sum().clamp_min(1) / (observed / observed.sum().clamp_min(1)).clamp_min(1e-12)
        weights = ratio[groups[positive]]
        self.sample_weight_ = torch.ones_like(p.targets)
        self.sample_weight_[positive] = weights / weights.mean().clamp_min(1e-12)

    def _step(self, p):
        self.optimizer_.zero_grad()
        logits, _ = self._forward(p)
        m = self.method_params
        weights = self.sample_weight_ if m["debias_method"] in {"uge-w", "uge-c"} else None
        supervised = F.binary_cross_entropy_with_logits(logits, p.targets, weight=weights)
        groups = self._fair_groups(p)
        rates = torch.stack([logits[groups == g].sigmoid().mean() for g in torch.unique(groups)])
        fair = rates.var(unbiased=False)
        coeff = m["reg_weight"] if m["debias_method"] in {"uge-r", "uge-c"} else 0
        loss = supervised + coeff * fair
        loss.backward()
        self.optimizer_.step()
        return {"loss": loss.item(), "supervised_loss": supervised.item(), "regularization_loss": fair.item()}
