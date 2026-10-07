import torch

from ..base import BaseEstimator, TransformerMixin
from ..data import as_graph, clone_graph, num_nodes


class GraphTransformer(BaseEstimator, TransformerMixin):
    """Transductive transformations: return a fresh graph with learned edits."""

    def _remember(self, X):
        self.is_fitted_ = False
        self.graph_ = clone_graph(as_graph(X))
        self.n_nodes_in_ = num_nodes(self.graph_)
        raw = getattr(self.graph_, "x", None)
        self.n_features_in_ = 0 if raw is None else raw.shape[1]

    def _check_graph(self, X):
        self._check_fitted()
        if X is None:
            return
        graph = as_graph(X)
        equal = num_nodes(graph) == self.n_nodes_in_ and torch.equal(torch.as_tensor(graph.edge_index).cpu(), torch.as_tensor(self.graph_.edge_index).cpu())
        x1, x2 = getattr(graph, "x", None), getattr(self.graph_, "x", None)
        equal = equal and ((x1 is None and x2 is None) or (x1 is not None and x2 is not None and torch.equal(torch.as_tensor(x1).cpu(), torch.as_tensor(x2).cpu())))
        if not equal:
            raise ValueError("This transformer learns edits for one graph. Refit to transform a different graph.")

    def transform(self, X=None):
        self._check_graph(X)
        return clone_graph(self.transformed_graph_)

    def get_embeddings(self):
        self._check_fitted()
        if not hasattr(self, "embedding_"):
            raise AttributeError("This method returns a graph rather than embeddings.")
        return self.embedding_.copy()
