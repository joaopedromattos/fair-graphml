"""Unified graph transformer using the repository's FairEGM components."""
import numpy as np
import torch

from ._base import GraphTransformer
from ..data import clone_graph, edge_tensor, sensitive_tensor
from ..nn import adjacency, normalize
from .._vendor.fairegm.torch_layers.cfo_graph_conv import CFOGraphConv
from .._vendor.fairegm.torch_layers.few_graph_conv import FEWGraphConv
from .._vendor.fairegm.torch_layers.gfo_graph_conv import GFOGraphConv
from .._vendor.fairegm.torch_models.fair_model import FairModel, NodeDense
from .._vendor.fairegm.torch_models.autoencoders import ReconstructionModel, ReconstructionVariationalModel
from .._vendor.fairegm.torch_models.losses import (build_reconstruction_loss,
    build_reconstruction_loss_vgae, build_dp_link_divergence_loss_vgae, dp_link_divergence_loss)


class FairEGM(GraphTransformer):
    """FairEGM with GFO, CFO, or FEW layers and alternating optimizers.

    fit_transform returns a graph whose x contains the fair embeddings; edges
    remain the supplied training structure. Learned internal fair adjacency is
    exposed as learned_adjacency_. Layer implementations and losses are copied
    from the repo's PyTorch port, with native torch DenseGraphConv algebra.
    """

    def __init__(self, mode="gfo", embedding_dim=32, epochs=100, lr=0.001,
                 lambda_epochs=1, lambda_param=1.0, vgae=False, layer_kwargs=None,
                 device="cpu", random_state=0, max_dense_nodes=5000):
        self.mode, self.embedding_dim = mode, embedding_dim
        self.epochs, self.lr = epochs, lr
        self.lambda_epochs, self.lambda_param = lambda_epochs, lambda_param
        self.vgae, self.layer_kwargs = vgae, layer_kwargs
        self.device, self.random_state = device, random_state
        self.max_dense_nodes = max_dense_nodes

    def fit(self, X, y=None, *, sensitive_features=None):
        self._remember(X)
        if self.mode not in {"gfo", "cfo", "few"}:
            raise ValueError("mode must be 'gfo', 'cfo', or 'few'.")
        n, f = self.n_nodes_in_, self.n_features_in_
        if n > self.max_dense_nodes:
            raise ValueError("FairEGM uses dense adjacency; increase max_dense_nodes explicitly.")
        if not f:
            raise ValueError("FairEGM requires data.x.")
        if min(self.embedding_dim, self.epochs, self.lambda_epochs) < 1 or self.lr <= 0 or self.lambda_param < 0:
            raise ValueError("Dimensions and epoch counts must be positive, lr > 0, and lambda_param >= 0.")
        x = torch.as_tensor(self.graph_.x, device=self.device).float().clone()
        if not torch.isfinite(x).all():
            raise ValueError("Features must be finite.")
        edges = edge_tensor(self.graph_.edge_index, n, self.device)
        a = adjacency(edges, n, dense=True)
        norm_a = normalize(a + torch.eye(n, device=self.device))
        sens, groups = sensitive_tensor(self.graph_, sensitive_features, self.device)
        attrs = torch.nn.functional.one_hot(sens, len(groups)).float().unsqueeze(0)
        positive = a.sum().item()
        pos_weight = (n * n - positive) / max(positive, 1)
        cuda = torch.device(self.device)
        devices = [cuda.index or torch.cuda.current_device()] if cuda.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.random_state)
            kwargs = dict(self.layer_kwargs or {})
            if self.mode == "gfo":
                fair = GFOGraphConv(n, f, **kwargs)
            elif self.mode == "cfo":
                fair = CFOGraphConv(n, f, kwargs.pop("num_additions", n), **kwargs)
            else:
                fair = FEWGraphConv(n, **kwargs)
            dense = NodeDense(f, self.embedding_dim)
            if self.vgae:
                task = ReconstructionVariationalModel(self.embedding_dim, self.embedding_dim)
                factor = n * n / max((n * n - positive) * 2, 1)
                task_loss = build_reconstruction_loss_vgae(pos_weight, factor, n, self.embedding_dim)
                fair_loss = build_dp_link_divergence_loss_vgae(self.embedding_dim)
            else:
                task = ReconstructionModel(self.embedding_dim, self.embedding_dim)
                task_loss = build_reconstruction_loss(pos_weight)
                fair_loss = dp_link_divergence_loss
            self.model_ = FairModel(fair, dense, task, torch.device(self.device))
            self.model_.compile(torch.optim.Adam(list(dense.parameters()) + list(task.parameters()), lr=self.lr),
                                torch.optim.Adam(fair.parameters(), lr=self.lr), task_loss, fair_loss)
            history = self.model_.fit(x.unsqueeze(0), norm_a.unsqueeze(0), a.unsqueeze(0), attrs,
                                      self.epochs, self.lambda_epochs, self.lambda_param)
            self.history_ = [{"epoch": i + 1, "task_loss": history["task loss"][i], "fairness_loss": history["fair loss"][i]} for i in range(self.epochs)]
            z, fair_a = self.model_.predict_embeddings(x.unsqueeze(0), norm_a.unsqueeze(0))
            self.embedding_ = z.squeeze(0).numpy().copy()
            self.learned_adjacency_ = fair_a.squeeze(0).numpy().copy()
            if not np.isfinite(self.embedding_).all() or not all(np.isfinite(list(h.values())).all() for h in self.history_):
                raise FloatingPointError("FairEGM produced non-finite training values.")
            self.transformed_graph_ = clone_graph(self.graph_)
            self.transformed_graph_.x = torch.as_tensor(self.embedding_, device=self.graph_.x.device).clone()
        self.is_fitted_ = True
        return self


class GFO(FairEGM):
    """FairEGM with the group fairness optimization layer."""
    def __init__(self, **kwargs):
        mode = kwargs.pop("mode", "gfo")
        if mode != "gfo":
            raise ValueError("GFO requires mode='gfo'.")
        super().__init__(mode=mode, **kwargs)


class CFO(FairEGM):
    """FairEGM with the community fairness optimization layer."""
    def __init__(self, **kwargs):
        mode = kwargs.pop("mode", "cfo")
        if mode != "cfo":
            raise ValueError("CFO requires mode='cfo'.")
        super().__init__(mode=mode, **kwargs)


class FEW(FairEGM):
    """FairEGM with the fair edge weighting layer."""
    def __init__(self, **kwargs):
        mode = kwargs.pop("mode", "few")
        if mode != "few":
            raise ValueError("FEW requires mode='few'.")
        super().__init__(mode=mode, **kwargs)
