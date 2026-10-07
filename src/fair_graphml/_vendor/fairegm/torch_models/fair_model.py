from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
def tqdm(iterable):
    return iterable

def _get_activation(name: str):
    if name is None or name == 'linear':
        return None
    name = name.lower()
    if not hasattr(F, name):
        raise ValueError(f"Unsupported activation '{name}'.")
    return getattr(F, name)


class NodeDense(nn.Module):
    """Applies the same dense transformation to every node independently."""

    def __init__(self, in_features: int, out_features: int, activation: str = 'relu'):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = _get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        if self.activation is not None:
            out = self.activation(out)
        return out


class FairModel(nn.Module):
    def __init__(
        self,
        fair_layer: nn.Module,
        dense_layer: nn.Module,
        task_model: nn.Module,
        device: torch.device,
    ):
        super().__init__()
        self.fair_layer = fair_layer
        self.dense_layer = dense_layer
        self.task_model = task_model
        self.device = device
        self.to(device)

        self.compiled = False

    def forward(self, nodes: Tensor, edges: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        fair_conv, fair_edges = self.fair_layer(nodes, edges)
        fair_nodes = self.dense_layer(fair_conv)
        logits, embeddings = self.task_model(fair_nodes, fair_edges)
        return logits, embeddings, fair_edges

    def compile(
        self,
        task_optimizer: torch.optim.Optimizer,
        fair_optimizer: torch.optim.Optimizer,
        task_loss: Callable[[Tensor, Tensor], Tensor],
        fair_loss: Callable[[Tensor, Tensor], Tensor],
    ) -> None:
        self.task_optimizer = task_optimizer
        self.fair_optimizer = fair_optimizer
        self.task_loss = task_loss
        self.fair_loss = fair_loss
        self.compiled = True

    def _set_requires_grad(self, module: Iterable[nn.Module] | nn.Module, value: bool) -> None:
        if isinstance(module, (list, tuple, set)):
            modules = module
        else:
            modules = [module]
        for mod in modules:
            for param in mod.parameters():
                param.requires_grad_(value)

    def fit(
        self,
        nodes: Tensor,
        edges: Tensor,
        target: Tensor,
        sensitive_attributes: Tensor,
        epochs: int,
        lambda_epochs: int = 1,
        lambda_param: float = 1.0,
    ) -> Dict[str, list]:
        assert self.compiled, "Model must be compiled before use"

        nodes = nodes.to(self.device)
        edges = edges.to(self.device)
        target = target.to(self.device)
        sensitive_attributes = sensitive_attributes.to(self.device)

        history = {'task loss': [], 'fair loss': []}

        for _ in tqdm(range(epochs)):
            self.train()
            self._set_requires_grad(self.fair_layer, False)
            self._set_requires_grad([self.dense_layer, self.task_model], True)
            self.task_optimizer.zero_grad()
            logits, _, _ = self.forward(nodes, edges)
            task_loss = self.task_loss(target, logits)
            task_loss.backward()
            self.task_optimizer.step()

            fair_loss_value = torch.tensor(0.0, device=self.device)
            for _ in range(lambda_epochs):
                self._set_requires_grad(self.fair_layer, True)
                self._set_requires_grad([self.dense_layer, self.task_model], False)
                self.fair_optimizer.zero_grad()
                logits, _, _ = self.forward(nodes, edges)
                fair_loss_value = self.fair_loss(sensitive_attributes, logits)
                (lambda_param * fair_loss_value).backward()
                self.fair_optimizer.step()
                if hasattr(self.fair_layer, 'apply_constraints'):
                    self.fair_layer.apply_constraints()

            history['task loss'].append(task_loss.detach().cpu().item())
            history['fair loss'].append(fair_loss_value.detach().cpu().item())

        return history

    @torch.no_grad()
    def evaluate(
        self,
        nodes: Tensor,
        edges: Tensor,
        target: Tensor,
        sensitive_attributes: Tensor,
    ) -> Tuple[float, float]:
        nodes = nodes.to(self.device)
        edges = edges.to(self.device)
        target = target.to(self.device)
        sensitive_attributes = sensitive_attributes.to(self.device)
        self.eval()
        logits, _, _ = self.forward(nodes, edges)
        task_loss = self.task_loss(target, logits)
        fair_loss_value = self.fair_loss(sensitive_attributes, logits)
        return task_loss.item(), fair_loss_value.item()

    @torch.no_grad()
    def predict_embeddings(self, nodes: Tensor, edges: Tensor) -> Tuple[Tensor, Tensor]:
        nodes = nodes.to(self.device)
        edges = edges.to(self.device)
        self.eval()
        _, embeddings, fair_edges = self.forward(nodes, edges)
        return embeddings.cpu(), fair_edges.cpu()

    def get_fair_modifications(self):
        params = []
        for param in self.fair_layer.parameters():
            params.append(param.detach().cpu().clone())
        return params
