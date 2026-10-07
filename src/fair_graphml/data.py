"""Validation and split handling for PyG Data and single-graph datasets."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch


def as_graph(X):
    if hasattr(X, "edge_index"):
        graph = X
    else:
        try:
            if len(X) != 1:
                raise ValueError("Expected a single-graph dataset; pass dataset[i] for multi-graph datasets.")
            graph = X[0]
        except TypeError as exc:
            raise TypeError("Expected a PyG Data object or a dataset containing one graph.") from exc
    if getattr(graph, "edge_index", None) is None:
        raise ValueError("The graph must provide edge_index.")
    return graph


def clone_graph(graph):
    return graph.clone() if hasattr(graph, "clone") else copy.deepcopy(graph)


def num_nodes(graph):
    n = getattr(graph, "num_nodes", None)
    if n is None and getattr(graph, "x", None) is not None:
        n = len(graph.x)
    if n is None or n < 1:
        raise ValueError("Set data.num_nodes or provide data.x, including isolated nodes.")
    return int(n)


def edge_tensor(edges, n, device="cpu"):
    raw = torch.as_tensor(edges, device=device)
    if raw.ndim != 2 or raw.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if raw.numel() and (not torch.isfinite(raw).all() or not (raw == raw.long()).all()):
        raise ValueError("Node indices must be finite integers.")
    edges = raw.long()
    if edges.numel() and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("edge_index contains a node outside [0, num_nodes).")
    return edges


def node_mask(mask, n, device="cpu"):
    if mask is None:
        return torch.ones(n, dtype=torch.bool, device=device)
    raw = torch.as_tensor(mask, device=device)
    if raw.ndim != 1:
        raise ValueError("Masks must be one-dimensional; select a split column explicitly.")
    if raw.dtype == torch.bool:
        if len(raw) != n:
            raise ValueError("Boolean node mask must have num_nodes entries.")
        result = raw.clone()
    else:
        if raw.numel() and ((raw != raw.long()).any() or raw.min() < 0 or raw.max() >= n):
            raise ValueError("Node mask indices must be integers in [0, num_nodes).")
        result = torch.zeros(n, dtype=torch.bool, device=device)
        result[raw.long()] = True
    if not result.any():
        raise ValueError("The training mask cannot be empty.")
    return result


def sensitive_tensor(graph, sensitive=None, device="cpu", required=True):
    if sensitive is None:
        for key in ("sens", "sensitive_attributes", "sensitive", "s"):
            sensitive = getattr(graph, key, None)
            if sensitive is not None:
                break
    if sensitive is None:
        if required:
            raise ValueError("Provide sensitive_features=... or set data.sens (one group per node).")
        return None, None
    values = torch.as_tensor(sensitive, device=device)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or len(values) != num_nodes(graph):
        raise ValueError("Sensitive features must contain one categorical group per node.")
    if not torch.isfinite(values).all():
        raise ValueError("Sensitive features must be finite.")
    groups, codes = torch.unique(values, sorted=True, return_inverse=True)
    if len(groups) < 2 and required:
        raise ValueError("Fairness methods require at least two sensitive groups.")
    return codes.long(), groups


def undirected(edges, n):
    return torch.unique(torch.cat((edges, edges.flip(0)), dim=1), dim=1)


def sample_negative_edges(edges, n, count, *, random_state=0, directed=False):
    """Sample without replacement, excluding self links and all supplied positives."""
    pairs = edges.detach().cpu().t().tolist()
    key = (lambda u, v: (u, v)) if directed else (lambda u, v: (min(u, v), max(u, v)))
    occupied = {key(u, v) for u, v in pairs if u != v}
    available = n * (n - 1) // (1 if directed else 2) - len(occupied)
    count = min(int(count), available)
    if count < 1:
        raise ValueError("No non-edges are available. Supply edge_label_index and edge labels explicitly.")
    rng = np.random.default_rng(random_state)
    chosen = set()
    for _ in range(max(100, count * 30)):
        u, v = rng.integers(n, size=2).tolist()
        pair = key(u, v)
        if u != v and pair not in occupied:
            chosen.add(pair)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        # Dense small graphs exhaust rejection sampling; enumerate only as needed.
        for u in rng.permutation(n).tolist():
            for v in rng.permutation(n).tolist():
                pair = key(u, v)
                if u != v and pair not in occupied:
                    chosen.add(pair)
                if len(chosen) == count:
                    break
            if len(chosen) == count:
                break
    result = sorted(chosen)
    rng.shuffle(result)
    return torch.tensor(result, dtype=torch.long, device=edges.device).t().contiguous()


def edge_groups(sens, edges, mode="dyadic"):
    """Contiguous unordered group-pair IDs, or binary inter/intra membership."""
    a, b = sens[edges[0]], sens[edges[1]]
    if mode == "inter":
        return (a != b).long()
    if mode != "dyadic":
        raise ValueError("group_mode must be 'dyadic' or 'inter'.")
    k = int(sens.max()) + 1
    lo, hi = torch.minimum(a, b), torch.maximum(a, b)
    return lo * k - lo * (lo - 1) // 2 + hi - lo


@dataclass
class PreparedGraph:
    graph: object
    x: torch.Tensor
    edges: torch.Tensor
    weights: torch.Tensor | None
    targets: torch.Tensor
    target_edges: torch.Tensor | None
    train_mask: torch.Tensor
    sens: torch.Tensor | None
    sensitive_mask: torch.Tensor
    classes: torch.Tensor
    groups: torch.Tensor | None


def prepare_graph(X, y=None, *, task="link_prediction", sensitive_features=None,
                  train_mask=None, sensitive_train_mask=None, edge_label_index=None,
                  edge_splits=None, device="cpu", random_state=0, directed=False,
                  require_sensitive=False):
    graph = as_graph(X)
    n = num_nodes(graph)
    raw_x = getattr(graph, "x", None)
    x = torch.ones(n, 1, device=device) if raw_x is None else torch.as_tensor(raw_x, device=device).float().clone()
    if x.ndim != 2 or len(x) != n or not torch.isfinite(x).all():
        raise ValueError("data.x must be a finite [num_nodes, num_features] matrix.")
    edges = edge_tensor(graph.edge_index, n, device)
    weights = getattr(graph, "edge_weight", None)
    if weights is not None:
        weights = torch.as_tensor(weights, device=device).float().clone()
        if weights.shape != (edges.shape[1],) or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("edge_weight must contain one finite nonnegative value per edge.")
    mask = node_mask(train_mask if train_mask is not None else getattr(graph, "train_mask", None), n, device)
    sens_mask = node_mask(sensitive_train_mask, n, device) if sensitive_train_mask is not None else mask.clone()
    sens, groups = sensitive_tensor(graph, sensitive_features, device, require_sensitive)
    target_edges = None
    if task == "node_classification":
        raw_y = y if y is not None else getattr(graph, "y", None)
        if raw_y is None:
            raise ValueError("Node classification requires y or data.y.")
        raw_y = torch.as_tensor(raw_y, device=device).reshape(-1)
        if len(raw_y) != n or not torch.isfinite(raw_y).all():
            raise ValueError("Node labels must contain one finite value per node.")
        # Only training labels establish the class vocabulary.
        classes = torch.unique(raw_y[mask], sorted=True)
        targets = torch.full((n,), -1, dtype=torch.long, device=device)
        for i, label in enumerate(classes):
            targets[raw_y == label] = i
        if len(classes) < 2:
            raise ValueError("Node classification requires at least two training classes.")
    elif task == "link_prediction":
        classes = torch.tensor([0, 1], device=device)
        if edge_splits is not None:
            if edge_label_index is not None or y is not None:
                raise ValueError("Use edge_splits or edge_label_index/y, not both.")
            part = edge_splits["train"]
            positive = edge_tensor(torch.as_tensor(part["edge"]).t(), n, device)
            negative = edge_tensor(torch.as_tensor(part["edge_neg"]).t(), n, device)
            edges = positive if directed else undirected(positive, n)
            weights = None
            target_edges = torch.cat((positive, negative), dim=1)
            targets = torch.cat((torch.ones(positive.shape[1], device=device), torch.zeros(negative.shape[1], device=device)))
        else:
            target_edges = edge_label_index
            if target_edges is None:
                target_edges = getattr(graph, "edge_label_index", None)
            if target_edges is not None:
                target_edges = edge_tensor(target_edges, n, device)
                labels = y if y is not None else getattr(graph, "edge_label", None)
                if labels is None:
                    raise ValueError("Provide y or data.edge_label with edge_label_index.")
                targets = torch.as_tensor(labels, device=device).float().reshape(-1)
            else:
                if y is not None:
                    raise ValueError("Provide edge_label_index when passing link labels y.")
                positive = edges[:, edges[0] != edges[1]]
                if not directed:
                    positive = torch.sort(positive, dim=0).values
                positive = torch.unique(positive, dim=1)
                if positive.shape[1] == 0:
                    raise ValueError("Link training requires at least one positive edge.")
                negative = sample_negative_edges(edges, n, positive.shape[1], random_state=random_state, directed=directed)
                target_edges = torch.cat((positive, negative), dim=1)
                targets = torch.cat((torch.ones(positive.shape[1], device=device), torch.zeros(negative.shape[1], device=device)))
        if len(targets) != target_edges.shape[1] or len(targets) == 0 or not ((targets == 0) | (targets == 1)).all():
            raise ValueError("Link labels must be binary (0/1), with one label per candidate edge.")
    else:
        raise ValueError("task must be 'link_prediction' or 'node_classification'.")
    return PreparedGraph(graph, x, edges, weights, targets, target_edges, mask, sens, sens_mask, classes, groups)


def replace_structure(graph, edges, weights=None):
    """Clone a graph and invalidate edge-aligned fields made stale by rewiring."""
    result = clone_graph(graph)
    result.edge_index = edges.detach().to(graph.edge_index.device).clone()
    result.edge_weight = None if weights is None else weights.detach().to(graph.edge_index.device).clone()
    for name in ("edge_attr", "adj", "adj_t"):
        if getattr(result, name, None) is not None:
            setattr(result, name, None)
    return result
