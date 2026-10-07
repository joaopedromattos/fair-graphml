"""Structural link scores from the structural_heuristics experiments."""
import math

import networkx as nx
import numpy as np

from .base import BaseEstimator, TransformerMixin
from .data import as_graph, edge_tensor, num_nodes, prepare_graph


class CommonNeighbors(BaseEstimator, TransformerMixin):
    """Common-neighbor link scores, in original candidate order.

    fit records the training graph. predict/decision_function return raw
    structural scores, rather than calibrated probabilities or class labels.
    """

    def fit(self, X, y=None, *, edge_label_index=None, edge_splits=None):
        self.is_fitted_ = False
        p = prepare_graph(X, y, edge_label_index=edge_label_index, edge_splits=edge_splits)
        self.graph_ = nx.Graph()
        self.graph_.add_nodes_from(range(len(p.x)))
        self.graph_.add_edges_from((u, v) for u, v in p.edges.t().tolist() if u != v)
        self.candidate_edges_ = p.target_edges.detach().cpu().clone()
        self.n_nodes_in_ = len(p.x)
        self.neighbors_ = [set(self.graph_.neighbors(i)) for i in range(len(p.x))]
        self._cache_ = {}
        self.is_fitted_ = True
        return self

    def _score_pair(self, u, v):
        return float(len(self.neighbors_[u] & self.neighbors_[v]))

    def decision_function(self, X=None, *, edge_label_index=None):
        self._check_fitted()
        if X is not None:
            graph = as_graph(X)
            if num_nodes(graph) != self.n_nodes_in_:
                raise ValueError("Heuristic candidates must use the fitted node IDs.")
            if edge_label_index is None:
                edge_label_index = getattr(graph, "edge_label_index", None)
            if edge_label_index is None:
                edge_label_index = graph.edge_index
        edges = self.candidate_edges_ if edge_label_index is None else edge_tensor(edge_label_index, self.n_nodes_in_)
        return np.array([self._score_pair(u, v) for u, v in edges.t().tolist()], dtype=float)

    def predict(self, X=None, *, edge_label_index=None):
        return self.decision_function(X, edge_label_index=edge_label_index)

    def transform(self, X=None):
        return self.decision_function(X).reshape(-1, 1)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform()


class AdamicAdar(CommonNeighbors):
    """Sum of inverse log-degree over shared neighbors. log_base defaults to e.

    Use log_base=10 to match the structural_heuristics notebook's convention.
    """

    def __init__(self, log_base=math.e):
        self.log_base = log_base

    def fit(self, X, y=None, **fit_params):
        if not np.isfinite(self.log_base) or self.log_base <= 1:
            raise ValueError("log_base must be finite and greater than one.")
        return super().fit(X, y, **fit_params)

    def _score_pair(self, u, v):
        return sum(math.log(self.log_base) / math.log(len(self.neighbors_[w]))
                   for w in self.neighbors_[u] & self.neighbors_[v] if len(self.neighbors_[w]) > 1)


class PersonalizedPageRank(CommonNeighbors):
    """PageRank score at the target, personalized and cached per source."""

    def __init__(self, alpha=0.85, tol=1e-7, max_iter=1000):
        self.alpha, self.tol, self.max_iter = alpha, tol, max_iter

    def fit(self, X, y=None, **fit_params):
        if not 0 < self.alpha < 1 or self.tol <= 0 or self.max_iter < 1:
            raise ValueError("Require 0 < alpha < 1, tol > 0, and max_iter >= 1.")
        return super().fit(X, y, **fit_params)

    def _score_pair(self, u, v):
        if u not in self._cache_:
            self._cache_[u] = nx.pagerank(self.graph_, alpha=self.alpha,
                                        personalization={u: 1.0}, tol=self.tol,
                                        max_iter=self.max_iter)
        return self._cache_[u][v]


CN, AA, PPR = CommonNeighbors, AdamicAdar, PersonalizedPageRank
__all__ = ["CommonNeighbors", "AdamicAdar", "PersonalizedPageRank", "CN", "AA", "PPR"]
