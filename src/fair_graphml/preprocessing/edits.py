"""PyG graph transformer around the original EDITS optimizer."""
import numpy as np
import torch

from ._base import GraphTransformer
from .._vendor.edits import EDITS as _EDITS
from ..data import edge_tensor, node_mask, replace_structure, sensitive_tensor
from ..nn import adjacency


class EDITS(GraphTransformer):
    """Learn feature scales and a debiased adjacency with a Wasserstein critic.

    The original adversarial/proximal optimization is used in float32 on the
    selected device. Feature truncation defaults to zero; set truncation=4 to
    reproduce the repo experiment's final feature removal. ``threshold``
    determines which learned adjacency values are exported as weighted edges.
    """

    def __init__(self, epochs=100, lr=0.003, weight_decay=1e-7, adj_lambda=0.1,
                 layer_threshold=2, dropout=0.1, alpha=0.05, beta=0.1,
                 truncation=0, threshold=0.0, device="cpu", random_state=0,
                 max_dense_nodes=5000, batch_size=None):
        self.epochs, self.lr, self.weight_decay = epochs, lr, weight_decay
        self.adj_lambda, self.layer_threshold, self.dropout = adj_lambda, layer_threshold, dropout
        self.alpha, self.beta = alpha, beta
        self.truncation, self.threshold = truncation, threshold
        self.device, self.random_state = device, random_state
        self.max_dense_nodes, self.batch_size = max_dense_nodes, batch_size

    def fit(self, X, y=None, *, sensitive_features=None, train_mask=None):
        self._remember(X)
        n = self.n_nodes_in_
        if n > self.max_dense_nodes:
            raise ValueError("EDITS learns a dense adjacency; increase max_dense_nodes explicitly.")
        if not isinstance(self.epochs, int) or self.epochs < 1 or self.lr <= 0:
            raise ValueError("epochs must be a positive integer and lr must be positive.")
        if not isinstance(self.truncation, int) or not 0 <= self.truncation <= self.n_features_in_:
            raise ValueError("truncation must be between zero and the feature count.")
        if self.batch_size is not None and (not isinstance(self.batch_size, int) or self.batch_size < 2):
            raise ValueError("batch_size must be at least two, or None.")
        raw = getattr(self.graph_, "x", None)
        if raw is None:
            raise ValueError("EDITS requires data.x.")
        x = torch.as_tensor(raw, device=self.device).float().clone()
        if not torch.isfinite(x).all():
            raise ValueError("Features must be finite.")
        x_norm = x / x.norm(dim=0).clamp_min(1e-12)
        sens, groups = sensitive_tensor(self.graph_, sensitive_features, self.device)
        if len(groups) != 2:
            raise ValueError("The original EDITS critic supports exactly two sensitive groups.")
        mask = node_mask(train_mask if train_mask is not None else getattr(self.graph_, "train_mask", None), n, self.device)
        idx = mask.nonzero().flatten()
        if len(torch.unique(sens[idx])) != 2:
            raise ValueError("The EDITS training mask must include both sensitive groups.")
        edges = edge_tensor(self.graph_.edge_index, n, self.device)
        a = adjacency(edges, n, dense=True)
        cuda = torch.device(self.device)
        devices = [cuda.index or torch.cuda.current_device()] if cuda.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.random_state)
            self.model_ = _EDITS(x, self.lr, self.weight_decay, adj_lambda=self.adj_lambda,
                                 layer_threshold=self.layer_threshold, dropout=self.dropout).to(self.device)
            self.history_ = []
            for epoch in range(self.epochs):
                # Stratify feature-update batches to avoid empty critic groups.
                if self.batch_size is None:
                    self.model_.optimize(a, x_norm, idx, sens, epoch, self.lr, self.alpha, self.beta)
                else:
                    self._batched_step(a, x_norm, idx, sens, epoch)
                self.model_.eval()
                with torch.no_grad():
                    edited_a, edited_x, critic, _, _ = self.model_(a, x_norm)
                    gap = critic[idx[sens[idx] == 1]].mean() - critic[idx[sens[idx] == 0]].mean()
                if not torch.isfinite(edited_a).all() or not torch.isfinite(edited_x).all():
                    raise FloatingPointError("EDITS produced non-finite graph values.")
                self.history_.append({"epoch": epoch + 1, "critic_gap": gap.item()})
            self.feature_scale_ = self.model_.x_debaising.s.detach().cpu().numpy().copy()
            # Export scales in the original feature units.
            output_x = x * self.model_.x_debaising.s.detach()
            if self.truncation:
                output_x[:, torch.argsort(self.model_.x_debaising.s)[:self.truncation]] = 0
            edited_a = edited_a.detach().clone()
            edited_a.fill_diagonal_(0)
            target_edges = (edited_a > self.threshold).nonzero().t()
            self.learned_adjacency_ = edited_a.cpu().numpy().copy()
            self.transformed_graph_ = replace_structure(self.graph_, target_edges, edited_a[target_edges[0], target_edges[1]])
            self.transformed_graph_.x = output_x.detach().to(self.graph_.x.device).clone()
        self.is_fitted_ = True
        return self

    def _batched_step(self, a, x, idx, sens, epoch):
        model = self.model_
        model.train()
        if epoch == 0:
            model.adj_renew.fit(a, self.lr)
        model.fc.requires_grad_(False)
        halves = [idx[sens[idx] == i] for i in range(2)]
        half_size = max(1, self.batch_size // 2)
        chunks = max((len(v) + half_size - 1) // half_size for v in halves)
        for i in range(chunks):
            # Cycle the smaller group to keep each critic batch well-defined.
            batch = torch.cat([v[(torch.arange(half_size, device=v.device) + i * half_size) % len(v)] for v in halves])
            model.optimizer_G.zero_grad()
            _, xd, critic, _, _ = model(a, x)
            adv = -(critic[batch[sens[batch] == 1]].mean() - critic[batch[sens[batch] == 0]].mean())
            loss = self.alpha * (xd[batch] - x[batch]).norm() + self.beta * adv
            loss.backward()
            model.optimizer_G.step()
            model.optimizer_feature_l1.step()
        _, xd, critic, _, _ = model(a, x)
        adv = -(critic[idx[sens[idx] == 1]].mean() - critic[idx[sens[idx] == 0]].mean())
        model.adj_renew.train_adj(xd, a, adv, epoch, self.lr)
        with torch.no_grad():
            model.x_debaising.s.clamp_(0, 1)
        model.fc.requires_grad_(True)
        for _ in range(8):
            model.optimizer_A.zero_grad()
            _, _, critic, _, _ = model(a, x)
            loss = critic[idx[sens[idx] == 1]].mean() - critic[idx[sens[idx] == 0]].mean()
            loss.backward()
            model.optimizer_A.step()
            with torch.no_grad():
                for p in model.fc.parameters():
                    p.clamp_(-0.02, 0.02)


class EDITSBatched(EDITS):
    """EDITS with stratified feature-update batches (dense adjacency retained)."""
    def __init__(self, *, batch_size=32, **kwargs):
        super().__init__(batch_size=batch_size, **kwargs)
