"""Graph layers using native torch sparse tensors; no compiled extensions."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def adjacency(edges, n, weights=None, *, dense=False, self_loops=False):
    weights = torch.ones(edges.shape[1], device=edges.device) if weights is None else torch.as_tensor(weights, device=edges.device).float()
    if self_loops:
        present = torch.zeros(n, dtype=torch.bool, device=edges.device)
        present[edges[0, edges[0] == edges[1]]] = True
        missing = torch.arange(n, device=edges.device)[~present]
        edges = torch.cat((edges, torch.stack((missing, missing))), dim=1)
        weights = torch.cat((weights, torch.ones(len(missing), device=weights.device, dtype=weights.dtype)))
    # A[target, source] gives PyG source-to-target message passing.
    a = torch.sparse_coo_tensor(edges.flip(0), weights, (n, n)).coalesce()
    return a.to_dense() if dense else a


def normalize(a):
    if a.layout == torch.strided:
        degree = a.sum(1).clamp_min(1e-12).rsqrt()
        return degree[:, None] * a * degree[None, :]
    a = a.coalesce()
    degree = torch.sparse.sum(a, dim=1).to_dense().clamp_min(1e-12).rsqrt()
    i = a.indices()
    return torch.sparse_coo_tensor(i, a.values() * degree[i[0]] * degree[i[1]], a.shape).coalesce()


class GraphEncoder(nn.Module):
    def __init__(self, inputs, hidden, outputs, dropout=0.1, kind="gcn"):
        super().__init__()
        if kind not in {"gcn", "sage", "gin", "mlp"}:
            raise ValueError("encoder must be 'gcn', 'sage', 'gin', or 'mlp'.")
        self.kind, self.dropout = kind, dropout
        self.lin1 = nn.Linear(inputs * (2 if kind == "sage" else 1), hidden)
        self.lin2 = nn.Linear(hidden * (2 if kind == "sage" else 1), outputs)
        if kind == "gin":
            self.eps = nn.Parameter(torch.zeros(2))

    def forward(self, x, a):
        if self.kind == "gcn":
            a = normalize(a)
        elif self.kind == "sage":
            if a.layout == torch.strided:
                a = a / a.sum(1, keepdim=True).clamp_min(1)
            else:
                a = a.coalesce()
                degree = torch.sparse.sum(a, 1).to_dense().clamp_min(1)
                a = torch.sparse_coo_tensor(a.indices(), a.values() / degree[a.indices()[0]], a.shape).coalesce()
        for i, linear in enumerate((self.lin1, self.lin2)):
            if self.kind == "gcn":
                x = linear(a @ x)
            elif self.kind == "sage":
                x = linear(torch.cat((x, a @ x), dim=-1))
            elif self.kind == "gin":
                x = linear((1 + self.eps[i]) * x + a @ x)
            else:
                x = linear(x)
            if i == 0:
                x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x


class Predictor(nn.Module):
    def __init__(self, dim, classes, task, decoder="dot"):
        super().__init__()
        self.task, self.decoder = task, decoder
        if task == "node_classification":
            self.head = nn.Linear(dim, classes)
        elif decoder == "mlp":
            self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        elif decoder != "dot":
            raise ValueError("decoder must be 'dot' or 'mlp'.")

    def forward(self, z, edges=None):
        if self.task == "node_classification":
            return self.head(z)
        product = z[edges[0]] * z[edges[1]]
        return self.head(product).squeeze(-1) if self.decoder == "mlp" else product.sum(-1)


class GraphModel(nn.Module):
    def __init__(self, inputs, hidden, dim, classes, task, dropout, encoder, decoder):
        super().__init__()
        self.encoder = GraphEncoder(inputs, hidden, dim, dropout, encoder)
        self.predictor = Predictor(dim, classes, task, decoder)

    def forward(self, x, a, edges=None):
        z = self.encoder(x, a)
        return self.predictor(z, edges), z


def straight_through_bernoulli(logits, temperature=1.0):
    u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
    soft = torch.sigmoid((logits + u.log() - (-u).log1p()) / temperature)
    return (soft > 0.5).to(soft.dtype).detach() - soft.detach() + soft
