import networkx as nx
import torch

from ._base import GraphTransformer
from ..data import edge_tensor, replace_structure, sensitive_tensor
from .._vendor.fairlp import fairlp


class FairLP(GraphTransformer):
    """Equalize intra/inter-group edge density by common-neighbor edits.

    ``mode`` is 'avg', 'min', or 'max'. ``protected_group`` selects an original
    sensitive-group value. The original algorithm is retained verbatim except
    for disabling progress output. It enumerates potential edges (O(N²)).
    """

    def __init__(self, mode="avg", protected_group=1, max_nodes=5000):
        self.mode = mode
        self.protected_group = protected_group
        self.max_nodes = max_nodes

    def fit(self, X, y=None, *, sensitive_features=None):
        self._remember(X)
        n = self.n_nodes_in_
        if n > self.max_nodes:
            raise ValueError("FairLP enumerates node pairs; increase max_nodes explicitly for this graph.")
        sens, groups = sensitive_tensor(self.graph_, sensitive_features)
        matches = (groups == self.protected_group).nonzero().flatten()
        if len(matches) != 1:
            raise ValueError("protected_group must be one of the sensitive group values.")
        edges = edge_tensor(self.graph_.edge_index, n)
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from((a, b) for a, b in edges.t().tolist() if a != b)
        protected = (sens == matches.item()).nonzero().flatten().tolist()
        graph, added, removed = fairlp(graph, protected, self.mode)
        pairs = sorted(graph.edges())
        result = torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs else torch.empty(2, 0, dtype=torch.long)
        result = torch.cat((result, result.flip(0)), dim=1)
        self.added_edges_, self.removed_edges_ = sorted(added), sorted(removed)
        self.transformed_graph_ = replace_structure(self.graph_, result)
        self.is_fitted_ = True
        return self
