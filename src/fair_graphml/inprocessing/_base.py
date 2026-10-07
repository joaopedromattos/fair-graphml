"""Shared supervised estimator lifecycle."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..base import BaseEstimator, TransformerMixin
from ..data import as_graph, edge_groups, edge_tensor, num_nodes, prepare_graph
from ..nn import GraphModel, adjacency


class GraphEstimator(BaseEstimator, TransformerMixin):
    """Full-batch node classification or binary link prediction.

    Method-specific keyword parameters are validated against each class's
    ``method_defaults``. All outputs follow sklearn conventions (NumPy arrays).
    The input graph and global random state are preserved.
    """

    _estimator_type = "classifier"
    requires_sensitive = False
    method_defaults = {}

    def __init__(self, *, hidden_channels=32, embedding_dim=16, epochs=100,
                 lr=0.01, weight_decay=1e-5, dropout=0.1, encoder="gcn",
                 decoder="dot", task="link_prediction", device="cpu",
                 random_state=0, verbose=False, directed=False, **method_params):
        unknown = set(method_params) - self.method_defaults.keys()
        if unknown:
            raise TypeError(f"Unknown {type(self).__name__} parameters: {sorted(unknown)}. "
                            f"Method parameters: {sorted(self.method_defaults)}")
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.encoder = encoder
        self.decoder = decoder
        self.task = task
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.directed = directed
        self.method_params = dict(self.method_defaults, **method_params)

    def __sklearn_tags__(self):
        from sklearn.utils import ClassifierTags, Tags, TargetTags
        return Tags(estimator_type="classifier", target_tags=TargetTags(required=True),
                    transformer_tags=None, regressor_tags=None,
                    classifier_tags=ClassifierTags(), array_api_support=False,
                    no_validation=True, non_deterministic=False,
                    requires_fit=True, _skip_test=True)

    def _validate_params(self):
        for key in ("hidden_channels", "embedding_dim", "epochs"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if self.lr <= 0 or self.weight_decay < 0 or not 0 <= self.dropout < 1:
            raise ValueError("Require lr > 0, weight_decay >= 0, and 0 <= dropout < 1.")
        if not isinstance(self.random_state, int) or self.random_state < 0:
            raise ValueError("random_state must be a nonnegative integer.")

    def _build(self, p):
        return GraphModel(p.x.shape[1], self.hidden_channels, self.embedding_dim,
                          len(p.classes), self.task, self.dropout, self.encoder,
                          self.decoder).to(self.device)

    def _setup(self, p):
        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.lr,
                                           weight_decay=self.weight_decay)

    def _supervised_loss(self, logits, p):
        if self.task == "node_classification":
            return F.cross_entropy(logits[p.train_mask], p.targets[p.train_mask])
        return F.binary_cross_entropy_with_logits(logits, p.targets)

    def _forward(self, p):
        return self.model_(p.x, self.adjacency_, p.target_edges)

    def _step(self, p):
        self.optimizer_.zero_grad()
        logits, z = self._forward(p)
        loss = self._supervised_loss(logits, p)
        loss.backward()
        self.optimizer_.step()
        return {"loss": float(loss.detach())}

    def _fair_groups(self, p):
        if self.task == "link_prediction":
            return edge_groups(p.sens, p.target_edges, self.method_params.get("group_mode", "dyadic"))
        return p.sens[p.train_mask]

    def _fair_scores(self, logits, p):
        return logits if self.task == "link_prediction" else logits[p.train_mask].softmax(-1)[:, -1]

    def fit(self, X, y=None, *, sensitive_features=None, train_mask=None,
            sensitive_train_mask=None, edge_label_index=None, edge_splits=None):
        """Fit using training labels only and return self.

        ``edge_splits`` accepts this repo's OGB dictionaries, with [E, 2]
        ``edge`` and ``edge_neg`` arrays. Only the train part is used.
        Alternatively use PyG's edge_label_index/edge_label fields.
        """
        self.is_fitted_ = False
        self._validate_params()
        p = prepare_graph(X, y, task=self.task, sensitive_features=sensitive_features,
                          train_mask=train_mask, sensitive_train_mask=sensitive_train_mask,
                          edge_label_index=edge_label_index, edge_splits=edge_splits,
                          device=self.device, random_state=self.random_state,
                          directed=self.directed, require_sensitive=self.requires_sensitive)
        self._prepared_ = p
        self.n_features_in_ = p.x.shape[1]
        self.classes_ = p.classes.detach().cpu().numpy()
        self.sensitive_groups_ = None if p.groups is None else p.groups.detach().cpu().numpy()
        self.adjacency_ = adjacency(p.edges, len(p.x), p.weights, self_loops=True)
        cuda = torch.device(self.device)
        devices = [cuda.index or torch.cuda.current_device()] if cuda.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.random_state)
            self.model_ = self._build(p)
            self._setup(p)
            self.history_ = []
            for epoch in range(self.epochs):
                self.model_.train()
                values = self._step(p)
                if not all(np.isfinite(v) for v in values.values()):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}: {values}")
                self.history_.append({"epoch": epoch + 1, **values})
                if self.verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                    print(f"{type(self).__name__} epoch {epoch + 1}: {values}")
            self.model_.eval()
            with torch.no_grad():
                _, embeddings = self._forward(p)
                self.embedding_ = embeddings.detach().cpu().numpy().copy()
        self.is_fitted_ = True
        return self

    def _inference(self, X, edge_label_index):
        self._check_fitted()
        p = self._prepared_
        if X is None:
            x, a = p.x, self.adjacency_
            edges = p.target_edges if edge_label_index is None else edge_tensor(edge_label_index, len(x), self.device)
        else:
            graph = as_graph(X)
            n = num_nodes(graph)
            raw = getattr(graph, "x", None)
            x = torch.ones(n, 1, device=self.device) if raw is None else torch.as_tensor(raw, device=self.device).float()
            if x.ndim != 2 or x.shape[1] != self.n_features_in_ or not torch.isfinite(x).all():
                raise ValueError("Prediction features must be finite and match the fitted feature dimension.")
            message_edges = edge_tensor(graph.edge_index, n, self.device)
            a = adjacency(message_edges, n, getattr(graph, "edge_weight", None), self_loops=True)
            if edge_label_index is None:
                edge_label_index = getattr(graph, "edge_label_index", None)
            if self.task == "link_prediction":
                edges = edge_tensor(edge_label_index, n, self.device) if edge_label_index is not None else message_edges
            else:
                edges = None
        self.model_.eval()
        with torch.no_grad():
            return self._predict_tensors(x, a, edges)

    def _predict_tensors(self, x, a, edges):
        return self.model_(x, a, edges)

    def decision_function(self, X=None, *, edge_label_index=None, mask=None):
        """Return link logits or node class logits, before sigmoid/softmax."""
        logits, _ = self._inference(X, edge_label_index)
        result = logits.detach().cpu().numpy()
        return result if mask is None else result[np.asarray(torch.as_tensor(mask).cpu())]

    def predict_proba(self, X=None, *, edge_label_index=None, mask=None):
        logits = torch.as_tensor(self.decision_function(X, edge_label_index=edge_label_index, mask=mask))
        if self.task == "link_prediction":
            p = logits.sigmoid().numpy()
            return np.column_stack((1 - p, p))
        return logits.softmax(-1).numpy()

    def predict(self, X=None, *, edge_label_index=None, mask=None):
        """Return class labels; use predict_proba for ranking/link scores."""
        prob = self.predict_proba(X, edge_label_index=edge_label_index, mask=mask)
        return self.classes_[prob.argmax(-1)]

    def transform(self, X=None):
        """Return node embeddings, ordered by node index."""
        if X is None:
            self._check_fitted()
            return self.embedding_.copy()
        _, embeddings = self._inference(X, None)
        return embeddings.detach().cpu().numpy()

    def fit_transform(self, X, y=None, **fit_params):
        # Return representations of the training structure, including when
        # edge_splits replaced a full input graph with training-only edges.
        return self.fit(X, y, **fit_params).transform()

    def score(self, X=None, y=None, *, edge_label_index=None, mask=None):
        if y is None:
            if X is None:
                p = self._prepared_
                if self.task == "node_classification" and mask is None:
                    mask = p.train_mask.cpu().numpy()
                y = p.targets.cpu().numpy() if self.task == "link_prediction" else self.classes_[p.targets.cpu().numpy()]
            else:
                graph = as_graph(X)
                y = getattr(graph, "edge_label" if self.task == "link_prediction" else "y", None)
        if y is None:
            raise ValueError("Provide labels to score.")
        labels = np.asarray(torch.as_tensor(y).cpu()).reshape(-1)
        if mask is not None:
            labels = labels[np.asarray(torch.as_tensor(mask).cpu())]
        predictions = self.predict(X, edge_label_index=edge_label_index, mask=mask)
        if labels.shape != predictions.shape:
            raise ValueError("Labels and predictions must have matching shapes.")
        return float(np.mean(labels == predictions))
