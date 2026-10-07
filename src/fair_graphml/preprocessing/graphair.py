"""Learned graph augmentation with contrastive/adversarial reconstruction."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ._base import GraphTransformer
from ..data import prepare_graph, replace_structure
from ..nn import GraphEncoder, adjacency, normalize, straight_through_bernoulli


class _Augmenter(nn.Module):
    def __init__(self, inputs, hidden, temperature):
        super().__init__()
        self.encoder = GraphEncoder(inputs, hidden, hidden)
        self.edge_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, inputs))
        self.feature_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, inputs))
        self.temperature = temperature

    def forward(self, x, a, raw_a, sample=True):
        h = self.encoder(x, a)
        z = self.edge_head(h)
        logits = z @ z.t()
        probability = 0.5 * logits.sigmoid() + 0.5 * raw_a
        mask_logits = self.feature_head(h)
        if sample:
            sampled = straight_through_bernoulli(torch.logit(probability.clamp(1e-6, 1 - 1e-6)), self.temperature)
            mask = straight_through_bernoulli(mask_logits, self.temperature)
        else:
            sampled, mask = probability, mask_logits.sigmoid()
        sampled = sampled.triu(1)
        sampled = sampled + sampled.t()
        eye = torch.eye(len(x), device=x.device)
        return sampled + eye, x * mask, logits


class Graphair(GraphTransformer):
    """Graphair's learned edge/feature augmentation and InfoNCE objective.

    fit_transform returns a graph with learned edge probabilities as weights
    and masked features. get_embeddings returns contrastive node embeddings.
    Full-batch dense augmentation has quadratic memory cost.
    """

    def __init__(self, hidden_channels=32, embedding_dim=32, epochs=100, lr=0.001,
                 weight_decay=1e-5, alpha=0.1, beta=1.0, gamma=10.0, lam=1.0,
                 temperature=1.0, contrastive_temperature=0.07, adversary_steps=1,
                 threshold=0.5, device="cpu", random_state=0, max_dense_nodes=5000):
        self.hidden_channels, self.embedding_dim = hidden_channels, embedding_dim
        self.epochs, self.lr, self.weight_decay = epochs, lr, weight_decay
        self.alpha, self.beta, self.gamma, self.lam = alpha, beta, gamma, lam
        self.temperature, self.contrastive_temperature = temperature, contrastive_temperature
        self.adversary_steps, self.threshold = adversary_steps, threshold
        self.device, self.random_state, self.max_dense_nodes = device, random_state, max_dense_nodes

    def fit(self, X, y=None, *, sensitive_features=None, train_mask=None,
            edge_label_index=None, edge_splits=None):
        self._remember(X)
        if self.n_nodes_in_ > self.max_dense_nodes:
            raise ValueError("Graphair uses dense adjacency; increase max_dense_nodes explicitly.")
        if min(self.hidden_channels, self.embedding_dim, self.epochs, self.adversary_steps) < 1 or self.lr <= 0 or min(self.temperature, self.contrastive_temperature) <= 0 or not 0 <= self.threshold <= 1:
            raise ValueError("Require positive dimensions, epochs, lr and temperatures, and threshold in [0, 1].")
        p = prepare_graph(X, y, sensitive_features=sensitive_features, train_mask=train_mask,
                          edge_label_index=edge_label_index, edge_splits=edge_splits,
                          device=self.device, random_state=self.random_state, require_sensitive=True)
        raw_a = adjacency(p.edges, len(p.x), dense=True).clamp(0, 1)
        a = raw_a + torch.eye(len(p.x), device=self.device)
        cuda = torch.device(self.device)
        devices = [cuda.index or torch.cuda.current_device()] if cuda.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.random_state)
            self.model_ = nn.ModuleDict({
                "augmenter": _Augmenter(p.x.shape[1], self.hidden_channels, self.temperature),
                "encoder": GraphEncoder(p.x.shape[1], self.hidden_channels, self.embedding_dim),
                "projector": nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ELU(), nn.Linear(self.embedding_dim, self.embedding_dim)),
                "adversary_encoder": GraphEncoder(p.x.shape[1], self.hidden_channels, len(p.groups)),
            }).to(self.device)
            main = [v for name, v in self.model_.named_parameters() if not name.startswith("adversary_encoder.")]
            self.optimizer_ = torch.optim.Adam(main, lr=self.lr, weight_decay=self.weight_decay)
            adv = self.model_["adversary_encoder"]
            adv_optimizer = torch.optim.Adam(adv.parameters(), lr=self.lr)
            self.history_ = []
            for epoch in range(self.epochs):
                self.model_.train()
                aug_a, aug_x, logits = self.model_["augmenter"](p.x, a, raw_a)
                for _ in range(self.adversary_steps):
                    adv_optimizer.zero_grad()
                    pred_s = adv(aug_x.detach(), aug_a.detach())
                    adv_loss = F.cross_entropy(pred_s[p.sensitive_mask], p.sens[p.sensitive_mask])
                    adv_loss.backward()
                    adv_optimizer.step()
                adv.requires_grad_(False)
                self.optimizer_.zero_grad()
                z1 = self.model_["projector"](self.model_["encoder"](p.x, a))
                z2 = self.model_["projector"](self.model_["encoder"](aug_x, aug_a))
                z = F.normalize(torch.cat((z1, z2)), dim=-1)
                similarity = z @ z.t() / self.contrastive_temperature
                similarity = similarity.masked_fill(torch.eye(len(z), dtype=torch.bool, device=z.device), -torch.inf)
                targets = (torch.arange(len(z), device=z.device) + len(p.x)) % len(z)
                contrast = F.cross_entropy(similarity, targets)
                target = p.target_edges
                edge_loss = F.binary_cross_entropy_with_logits(logits[target[0], target[1]], p.targets)
                feature_loss = F.mse_loss(aug_x, p.x)
                pred_s = adv(aug_x, aug_a)
                adv_loss = F.cross_entropy(pred_s[p.sensitive_mask], p.sens[p.sensitive_mask])
                loss = self.beta * contrast + self.gamma * (edge_loss + self.lam * feature_loss) - self.alpha * adv_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Graphair produced a non-finite loss.")
                loss.backward()
                self.optimizer_.step()
                adv.requires_grad_(True)
                self.history_.append({"epoch": epoch + 1, "loss": loss.item(), "contrastive_loss": contrast.item(), "edge_loss": edge_loss.item(), "feature_loss": feature_loss.item(), "adversary_loss": adv_loss.item()})
            self.model_.eval()
            with torch.no_grad():
                aug_a, aug_x, _ = self.model_["augmenter"](p.x, a, raw_a, sample=False)
                self.embedding_ = self.model_["encoder"](aug_x, aug_a).cpu().numpy().copy()
                aug_a.fill_diagonal_(0)
                edge_index = (aug_a > self.threshold).nonzero().t()
                self.transformed_graph_ = replace_structure(self.graph_, edge_index, aug_a[edge_index[0], edge_index[1]])
                self.transformed_graph_.x = aug_x.detach().to(self.graph_.x.device if getattr(self.graph_, "x", None) is not None else "cpu").clone()
                self.learned_adjacency_ = aug_a.cpu().numpy().copy()
        self.is_fitted_ = True
        return self


GraphAir = Graphair
