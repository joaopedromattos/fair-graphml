"""Group-balanced and biased random walks with native torch skip-gram."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..base import BaseEstimator, TransformerMixin
from ..data import as_graph, clone_graph, edge_tensor, num_nodes, sensitive_tensor


class Node2Vec(BaseEstimator, TransformerMixin):
    """Second-order random-walk embeddings with negative-sampling skip-gram.

    Supports isolated nodes and arbitrary integer sensitive labels. No gensim,
    torch-cluster, or pyg-lib dependency is required. transform is transductive.
    """
    balanced = False

    def __init__(self, embedding_dim=64, walk_length=20, num_walks=5, context_size=5,
                 p=1.0, q=1.0, negative_samples=5, epochs=5, batch_size=256,
                 lr=0.01, device="cpu", random_state=0, directed=False):
        self.embedding_dim, self.walk_length, self.num_walks = embedding_dim, walk_length, num_walks
        self.context_size, self.p, self.q = context_size, p, q
        self.negative_samples, self.epochs, self.batch_size = negative_samples, epochs, batch_size
        self.lr, self.device, self.random_state, self.directed = lr, device, random_state, directed

    def fit(self, X, y=None, *, sensitive_features=None):
        self.is_fitted_ = False
        for key in ("embedding_dim", "walk_length", "num_walks", "context_size", "negative_samples", "epochs", "batch_size"):
            value = getattr(self, key)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if self.p <= 0 or self.q <= 0 or self.lr <= 0:
            raise ValueError("p, q, and lr must be positive.")
        self.graph_ = clone_graph(as_graph(X))
        n = num_nodes(self.graph_)
        edges = edge_tensor(self.graph_.edge_index, n)
        weights = getattr(self.graph_, "edge_weight", None)
        weights = np.ones(edges.shape[1]) if weights is None else np.asarray(torch.as_tensor(weights).cpu())
        if weights.shape != (edges.shape[1],) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("edge_weight must be finite, nonnegative, and aligned with edge_index.")
        self.neighbors_ = [dict() for _ in range(n)]
        for (u, v), weight in zip(edges.t().tolist(), weights.tolist()):
            if weight > 0:
                self.neighbors_[u][v] = weight
                if not self.directed:
                    self.neighbors_[v][u] = weight
        sens, groups = sensitive_tensor(self.graph_, sensitive_features, required=self.balanced)
        self.sensitive_codes_ = None if sens is None else sens.numpy()
        self.sensitive_groups_ = None if groups is None else groups.numpy()
        rng = np.random.default_rng(self.random_state)
        self.walks_ = []
        for _ in range(self.num_walks):
            for start in rng.permutation(n):
                walk = [int(start)]
                for _ in range(self.walk_length - 1):
                    probabilities = self._transition(walk[-1], walk[-2] if len(walk) > 1 else None)
                    if not probabilities:
                        break
                    nodes = list(probabilities)
                    walk.append(int(rng.choice(nodes, p=list(probabilities.values()))))
                self.walks_.append(walk)
        pairs = [(walk[i], walk[j]) for walk in self.walks_ for i in range(len(walk))
                 for j in range(max(0, i - self.context_size), min(len(walk), i + self.context_size + 1)) if i != j]
        if not pairs:
            raise ValueError("Walks contain no context pairs; provide edges and walk_length >= 2.")
        pairs = torch.tensor(pairs, dtype=torch.long, device=self.device)
        cuda = torch.device(self.device)
        devices = [cuda.index or torch.cuda.current_device()] if cuda.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.random_state)
            self.model_ = nn.Embedding(n, self.embedding_dim).to(self.device)
            nn.init.normal_(self.model_.weight, std=0.1)
            self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
            self.history_ = []
            for epoch in range(self.epochs):
                order = torch.randperm(len(pairs), device=self.device)
                total, count = 0.0, 0
                for indices in order.split(self.batch_size):
                    batch = pairs[indices]
                    u, v = self.model_(batch[:, 0]), self.model_(batch[:, 1])
                    negatives = torch.randint(n, (len(batch), self.negative_samples), device=self.device)
                    vn = self.model_(negatives)
                    loss = -F.logsigmoid((u * v).sum(-1)).mean() - F.logsigmoid(-(u[:, None, :] * vn).sum(-1)).mean()
                    self.optimizer_.zero_grad()
                    loss.backward()
                    self.optimizer_.step()
                    total += loss.item() * len(batch)
                    count += len(batch)
                self.history_.append({"epoch": epoch + 1, "loss": total / count})
            self.embedding_ = self.model_.weight.detach().cpu().numpy().copy()
        self.n_nodes_in_, self.n_features_out_ = n, self.embedding_dim
        if not np.isfinite(self.embedding_).all():
            raise FloatingPointError("Random-walk training produced non-finite embeddings.")
        self.is_fitted_ = True
        return self

    def _transition(self, current, previous=None):
        neighbors = self.neighbors_[current]
        if not neighbors:
            return {}
        nodes = list(neighbors)
        probabilities = np.array([neighbors[v] for v in nodes], dtype=float)
        if previous is not None:
            probabilities *= np.array([1 / self.p if v == previous else (1.0 if v in self.neighbors_[previous] else 1 / self.q) for v in nodes])
        if self.balanced:
            groups = self.sensitive_codes_[nodes]
            for group in np.unique(groups):
                mask = groups == group
                probabilities[mask] /= probabilities[mask].sum()
        probabilities /= probabilities.sum()
        return dict(zip(nodes, probabilities.tolist()))

    def transition_probabilities(self, node, previous=None):
        """Return the fitted walk's next-node probabilities (useful for auditing)."""
        self._check_fitted()
        if not 0 <= node < self.n_nodes_in_ or (previous is not None and not 0 <= previous < self.n_nodes_in_):
            raise ValueError("Node index is outside the fitted graph.")
        return self._transition(node, previous)

    def transform(self, X=None, *, nodes=None):
        self._check_fitted()
        if X is not None:
            graph = as_graph(X)
            if num_nodes(graph) != self.n_nodes_in_ or not torch.equal(torch.as_tensor(graph.edge_index).cpu(), torch.as_tensor(self.graph_.edge_index).cpu()):
                raise ValueError("Walk embeddings are transductive; refit for a different graph.")
        result = self.embedding_.copy()
        return result if nodes is None else result[np.asarray(torch.as_tensor(nodes).cpu())]


class FairWalk(Node2Vec):
    """Node2Vec walks that give equal probability to each available neighbor group.

    This activates the fair transition rule present in the repository's walk
    implementation; its current experiment fit loop had bypassed that rule.
    """
    balanced = True
