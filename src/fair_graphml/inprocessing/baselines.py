"""Neural common-neighbor and enclosing-subgraph link predictors."""
import networkx as nx
import torch
from torch import nn
from torch.nn import functional as F

from ._base import GraphEstimator
from ..nn import GraphEncoder, adjacency, normalize


class _NCNModel(nn.Module):
    def __init__(self, owner, p):
        super().__init__()
        dim, hidden = owner.embedding_dim, owner.hidden_channels
        self.encoder = GraphEncoder(p.x.shape[1], hidden, dim, owner.dropout, owner.encoder)
        self.xij = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.xcn = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.beta = nn.Parameter(torch.tensor(float(owner.method_params["beta"])))

    def forward(self, x, a, edges):
        z = self.encoder(x, a)
        index = a.nonzero().t() if a.layout == torch.strided else a.coalesce().indices()
        neighbors = [set() for _ in range(len(x))]
        for u, v in index.detach().cpu().t().tolist():
            if u != v:
                neighbors[u].add(v)
        common = []
        for u, v in edges.detach().cpu().t().tolist():
            ids = sorted(neighbors[u] & neighbors[v])
            common.append(z[ids].sum(0) if ids else z.new_zeros(z.shape[1]))
        if not common:
            return z.new_empty(0), z
        cn = torch.stack(common)
        logits = self.head(self.xij(z[edges[0]] * z[edges[1]]) + self.beta * self.xcn(cn)).squeeze(-1)
        return logits, z


class NCN(GraphEstimator):
    """Neural common-neighbor decoder: node products plus summed shared neighbors.

    Native sparse/Python intersection implementation of CNLinkPredictor's
    core algebra. Extra parameter: beta=1. No torch-sparse dependency.
    """
    method_defaults = {"beta": 1.0}

    def _build(self, p):
        if self.task != "link_prediction":
            raise ValueError("NCN supports link_prediction only.")
        return _NCNModel(self, p).to(self.device)


def enclosing_subgraph(x, edges, pair, num_hops=2):
    """Extract an enclosing graph and DRNL labels, removing the queried edge."""
    u, v = pair
    if u == v:
        raise ValueError("SEAL candidates must connect two distinct nodes.")
    graph = nx.Graph()
    graph.add_nodes_from(range(len(x)))
    graph.add_edges_from(edges.detach().cpu().t().tolist())
    nodes = {u, v}
    frontier = {u, v}
    for _ in range(num_hops):
        frontier = {neighbor for node in frontier for neighbor in graph.neighbors(node)} - nodes
        nodes.update(frontier)
    order = [u, v] + sorted(nodes - {u, v})
    local = {node: i for i, node in enumerate(order)}
    sub = nx.relabel_nodes(graph.subgraph(order).copy(), local)
    if sub.has_edge(0, 1):
        sub.remove_edge(0, 1)
    no_dst, no_src = sub.copy(), sub.copy()
    no_dst.remove_node(1)
    no_src.remove_node(0)
    dsrc = nx.single_source_shortest_path_length(no_dst, 0)
    ddst = nx.single_source_shortest_path_length(no_src, 1)
    labels = [1, 1]
    for i in range(2, len(order)):
        a, b = dsrc.get(i), ddst.get(i)
        if a is None or b is None:
            labels.append(0)
        else:
            d = a + b
            labels.append(1 + min(a, b) + (d // 2) * (d // 2 + d % 2 - 1))
    local_edges = list(sub.edges())
    ei = torch.tensor(local_edges, dtype=torch.long, device=x.device).t().contiguous() if local_edges else torch.empty(2, 0, dtype=torch.long, device=x.device)
    ei = torch.cat((ei, ei.flip(0)), dim=1)
    a = normalize(adjacency(ei, len(order), self_loops=True))
    return x[order], a, torch.tensor(labels, dtype=torch.long, device=x.device)


class _SEALModel(nn.Module):
    def __init__(self, owner, p):
        super().__init__()
        self.hops = owner.method_params["num_hops"]
        self.use_feature = owner.method_params["use_feature"]
        hidden = owner.hidden_channels
        self.label_embedding = nn.Embedding(owner.method_params["max_z"], hidden)
        inputs = hidden + (p.x.shape[1] if self.use_feature else 0)
        self.convs = nn.ModuleList([nn.Linear(inputs, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)])
        self.hidden = nn.Linear(hidden, owner.embedding_dim)
        self.head = nn.Linear(owner.embedding_dim, 1)
        self.dropout = owner.dropout

    def forward(self, x, a, edges):
        ei = a.nonzero().t() if a.layout == torch.strided else a.coalesce().indices()
        ei = ei[:, ei[0] != ei[1]]
        outputs = []
        for pair in edges.detach().cpu().t().tolist():
            sub_x, sub_a, labels = enclosing_subgraph(x, ei, pair, self.hops)
            if labels.max() >= self.label_embedding.num_embeddings:
                raise ValueError("DRNL label exceeds max_z; increase max_z.")
            h = self.label_embedding(labels)
            if self.use_feature:
                h = torch.cat((h, sub_x), dim=1)
            for i, conv in enumerate(self.convs):
                h = conv(sub_a @ h)
                if i < len(self.convs) - 1:
                    h = F.dropout(F.relu(h), self.dropout, self.training)
            pooled = F.dropout(F.relu(self.hidden(h[0] * h[1])), self.dropout, self.training)
            outputs.append(self.head(pooled).squeeze(-1))
        # SEAL learns subgraph representations, not global node embeddings.
        # Return the pooled edge representations separately via the estimator.
        logits = torch.stack(outputs) if outputs else x.new_empty(0)
        return logits, x


class SEAL(GraphEstimator):
    """Enclosing-subgraph GCN with double-radius node labeling and center pooling.

    Extra parameters: num_hops=2, max_z=1000, use_feature=False.
    This correctness-first implementation extracts subgraphs per candidate.
    SEAL has no global node embedding; transform returns subgraph edge logits.
    """
    method_defaults = {"num_hops": 2, "max_z": 1000, "use_feature": False}

    def _build(self, p):
        if self.task != "link_prediction":
            raise ValueError("SEAL supports link_prediction only.")
        if self.method_params["num_hops"] < 1 or self.method_params["max_z"] < 2:
            raise ValueError("Require num_hops >= 1 and max_z >= 2.")
        return _SEALModel(self, p).to(self.device)

    def transform(self, X=None):
        return self.decision_function(X).reshape(-1, 1)
