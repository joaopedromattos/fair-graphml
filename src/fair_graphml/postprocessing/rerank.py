"""Fair ranking transformers operating on arrays or PyG candidate edges."""
import math

import numpy as np
import torch

from ..base import BaseEstimator, TransformerMixin
from ..data import as_graph, edge_groups, edge_tensor, num_nodes, sensitive_tensor


class _Reranker(BaseEstimator, TransformerMixin):
    def _inputs(self, X, scores, sensitive_features, edge_label_index):
        is_graph = hasattr(X, "edge_index") or (not isinstance(X, (list, tuple, np.ndarray, torch.Tensor)) and hasattr(X, "__getitem__"))
        if is_graph:
            graph = as_graph(X)
            if scores is None:
                raise ValueError("Provide scores=... when reranking PyG graph candidates.")
            edges = edge_label_index if edge_label_index is not None else getattr(graph, "edge_label_index", None)
            if edges is None:
                raise ValueError("Provide edge_label_index or set data.edge_label_index.")
            edges = edge_tensor(edges, num_nodes(graph))
            sens, _ = sensitive_tensor(graph, sensitive_features)
            groups = edge_groups(sens, edges, self.group_mode).numpy()
        else:
            scores = X if scores is None else scores
            if sensitive_features is None:
                raise ValueError("Provide one sensitive_features group per ranking candidate.")
            groups = np.asarray(torch.as_tensor(sensitive_features).cpu()).reshape(-1)
        values = np.asarray(torch.as_tensor(scores).detach().cpu(), dtype=float).reshape(-1)
        if len(values) == 0 or len(values) != len(groups) or not np.isfinite(values).all():
            raise ValueError("Scores must be finite, nonempty, and aligned with sensitive_features.")
        if not np.isfinite(groups).all():
            raise ValueError("Candidate group labels must be finite.")
        return values, groups

    def fit(self, X, y=None, *, scores=None, sensitive_features=None, edge_label_index=None):
        self.is_fitted_ = False
        values, groups = self._inputs(X, scores, sensitive_features, edge_label_index)
        self.scores_, self.candidate_groups_ = values.copy(), groups.copy()
        self.groups_, counts = np.unique(groups, return_counts=True)
        if getattr(self, "distribution", None) is None:
            self.distribution_ = dict(zip(self.groups_.tolist(), (counts / counts.sum()).tolist()))
        else:
            if set(self.distribution) != set(self.groups_.tolist()):
                raise ValueError("distribution must specify exactly the observed candidate groups.")
            self.distribution_ = dict(self.distribution)
            probs = list(self.distribution_.values())
            if not all(np.isfinite(p) and p >= 0 for p in probs) or not np.isclose(sum(probs), 1):
                raise ValueError("distribution must be finite, nonnegative, and sum to one.")
        k = len(values) if self.k is None else self.k
        if not isinstance(k, int) or not 1 <= k <= len(values):
            raise ValueError("k must be between one and the number of candidates, or None.")
        self.ranking_ = np.asarray(self._rerank(values, groups, k), dtype=int)
        self.ranked_scores_ = values[self.ranking_].copy()
        self.is_fitted_ = True
        return self

    def transform(self, X=None, *, scores=None, sensitive_features=None, edge_label_index=None):
        self._check_fitted()
        if X is not None:
            if scores is None and sensitive_features is None and not hasattr(X, "edge_index"):
                candidate = np.asarray(torch.as_tensor(X).detach().cpu()).reshape(-1)
                if np.array_equal(candidate, self.scores_):
                    return self.ranking_.copy()
                if len(candidate) != len(self.candidate_groups_):
                    raise ValueError("New candidates require sensitive_features aligned with their scores.")
                sensitive_features = self.candidate_groups_
            if scores is not None or sensitive_features is not None:
                from copy import deepcopy
                return deepcopy(self).fit(X, scores=scores, sensitive_features=sensitive_features, edge_label_index=edge_label_index).ranking_.copy()
            raise ValueError("Provide scores when transforming graph candidates.")
        return self.ranking_.copy()

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform()

    def predict(self, X=None, **kwargs):
        """Return candidate indices in reranked order, never fabricated scores."""
        return self.transform(X, **kwargs)


class DetConstSort(_Reranker):
    """DetConstSort with score-ordered insertion and prefix count deadlines.

    k=None ranks all candidates. distribution maps group labels to desired
    proportions; None uses the candidate distribution. Returns original
    candidate indices. Requested prefix quotas are checked for feasibility.
    """

    def __init__(self, k=None, distribution=None, group_mode="dyadic"):
        self.k, self.distribution, self.group_mode = k, distribution, group_mode

    def _rerank(self, scores, groups, k):
        queues = {g: list(np.flatnonzero(groups == g)[np.argsort(-scores[groups == g], kind="stable")]) for g in self.groups_.tolist()}
        available = {g: len(v) for g, v in queues.items()}
        for g, probability in self.distribution_.items():
            if math.floor(k * probability + 1e-12) > available[g]:
                raise ValueError(f"Insufficient candidates from group {g!r} for requested top-{k} distribution.")
        ranking, deadlines = [], []
        minima = {g: 0 for g in queues}
        used = {g: 0 for g in queues}
        prefix = 0
        while len(ranking) < k:
            prefix += 1
            changed = [g for g in queues if math.floor(prefix * self.distribution_[g] + 1e-12) > minima[g] and used[g] < available[g]]
            changed.sort(key=lambda g: (-scores[queues[g][used[g]]], queues[g][used[g]]))
            for g in changed:
                candidate = queues[g][used[g]]
                ranking.append(candidate)
                deadlines.append(prefix)
                position = len(ranking) - 1
                while position > 0 and deadlines[position - 1] >= position + 1 and scores[ranking[position - 1]] < scores[ranking[position]]:
                    ranking[position - 1], ranking[position] = ranking[position], ranking[position - 1]
                    deadlines[position - 1], deadlines[position] = deadlines[position], deadlines[position - 1]
                    position -= 1
                used[g] += 1
            minima = {g: min(math.floor(prefix * self.distribution_[g] + 1e-12), available[g]) for g in queues}
            # Groups with desired mass zero are appended by relevance once all
            # positive-mass groups are exhausted; no unbounded loop is possible.
            if not changed and all(used[g] == available[g] or self.distribution_[g] == 0 for g in queues):
                remaining = [c for g in queues for c in queues[g][used[g]:]]
                remaining.sort(key=lambda c: (-scores[c], c))
                ranking.extend(remaining)
                break
            if not changed:
                # Jump to the next quota deadline instead of iterating through
                # potentially billions of prefixes for a tiny target mass.
                future = [math.ceil((minima[g] + 1) / self.distribution_[g]) for g in queues if used[g] < available[g] and self.distribution_[g] > 0]
                prefix = max(prefix, min(future) - 1)
        output = ranking[:k]
        for length in range(1, k + 1):
            for g, probability in self.distribution_.items():
                if np.count_nonzero(groups[output[:length]] == g) < math.floor(length * probability + 1e-12):
                    raise ValueError("Requested distribution has infeasible prefix quotas for these candidates.")
        return output


class EpsilonGreedy(_Reranker):
    """Seeded random swaps in a relevance-sorted ranking, as in EPSILONGREEDY."""

    def __init__(self, epsilon=0.1, k=None, random_state=0, group_mode="dyadic"):
        self.epsilon, self.k = epsilon, k
        self.random_state, self.group_mode = random_state, group_mode

    def _rerank(self, scores, groups, k):
        if not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1].")
        rng = np.random.RandomState(self.random_state)
        ranking = np.argsort(-scores, kind="stable")
        for i in range(len(ranking)):
            if rng.rand() <= self.epsilon and i < len(ranking) - 1:
                j = rng.randint(i + 1, len(ranking))
                ranking[i], ranking[j] = ranking[j], ranking[i]
        return ranking[:k]


DETCONSTSORT = DetConstSort
EPSILONGREEDY = EpsilonGreedy
