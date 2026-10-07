"""Group fairness metrics for predictions and ranked graph links."""
import numpy as np
import torch


def _array(x):
    return np.asarray(x.detach().cpu() if isinstance(x, torch.Tensor) else x)


def demographic_parity_difference(y_pred, sensitive_features, *, positive_label=1):
    """Maximum minus minimum group positive prediction rate."""
    pred, sens = _array(y_pred).reshape(-1), _array(sensitive_features).reshape(-1)
    if len(pred) != len(sens) or not len(pred):
        raise ValueError("Predictions and sensitive features must have matching nonempty lengths.")
    rates = [np.mean(pred[sens == g] == positive_label) for g in np.unique(sens)]
    return float(max(rates) - min(rates))


def equal_opportunity_difference(y_true, y_pred, sensitive_features, *, positive_label=1):
    """Group true-positive-rate gap; NaN when a group has no positives."""
    true, pred, sens = (_array(v).reshape(-1) for v in (y_true, y_pred, sensitive_features))
    if len(true) != len(pred) or len(pred) != len(sens) or not len(pred):
        raise ValueError("Labels, predictions, and sensitive features must have matching nonempty lengths.")
    rates = []
    for group in np.unique(sens):
        mask = (sens == group) & (true == positive_label)
        if not mask.any():
            return float("nan")
        rates.append(np.mean(pred[mask] == positive_label))
    return float(max(rates) - min(rates))


def ndkl(ranking, sensitive_features, distribution=None):
    """Normalized discounted KL(prefix || target) for candidate-index rankings.

    This evaluation metric follows pos_processing_methods/methods.py. The
    differentiable training loss uses the reverse KL, as in sinkhorn.py.
    """
    order = _array(ranking).reshape(-1)
    groups = _array(sensitive_features).reshape(-1)
    if len(order) == 0 or not np.issubdtype(order.dtype, np.integer) or np.any(order < 0) or np.any(order >= len(groups)) or len(np.unique(order)) != len(order):
        raise ValueError("ranking must contain distinct valid candidate indices.")
    labels, counts = np.unique(groups, return_counts=True)
    if distribution is None:
        target = counts / counts.sum()
    else:
        if set(distribution) != set(labels.tolist()):
            raise ValueError("distribution must cover exactly the candidate groups.")
        target = np.array([distribution[g] for g in labels])
        if np.any(target < 0) or not np.isfinite(target).all() or not np.isclose(target.sum(), 1):
            raise ValueError("distribution must be finite, nonnegative, and sum to one.")
    present = np.zeros(len(labels), dtype=float)
    weighted, normalizer = 0.0, 0.0
    for i, candidate in enumerate(order, 1):
        present[np.flatnonzero(labels == groups[candidate])[0]] += 1
        p = present / i
        mask = p > 0
        if np.any(target[mask] == 0):
            return float("inf")
        divergence = np.sum(p[mask] * np.log(p[mask] / target[mask]))
        weight = 1 / np.log2(i + 1)
        weighted += weight * divergence
        normalizer += weight
    return float(weighted / normalizer)
