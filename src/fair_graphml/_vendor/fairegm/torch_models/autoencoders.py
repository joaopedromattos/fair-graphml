from typing import Tuple

import torch
from torch import nn

from ..torch_layers.graph_cnn import GraphCNN
from ..torch_layers.link_reconstruction import LinkReconstruction


class BaseAutoEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, embedding_dim: int):
        super().__init__()
        self.conv1 = GraphCNN(in_channels, hidden_channels, activation='relu')
        self.conv2 = GraphCNN(hidden_channels, embedding_dim, activation='linear')
        self.decoder = LinkReconstruction()

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, adj = self.conv1(nodes, adj)
        embeddings, adj = self.conv2(hidden, adj)
        logits = self.decoder(embeddings)
        return logits, embeddings, adj


class VariationalAutoEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, embedding_dim: int):
        super().__init__()
        self.conv1 = GraphCNN(in_channels, hidden_channels, activation='relu')
        self.z_mean = GraphCNN(hidden_channels, embedding_dim, activation='linear')
        self.z_log_std = GraphCNN(hidden_channels, embedding_dim, activation='linear')
        self.decoder = LinkReconstruction()

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, adj = self.conv1(nodes, adj)
        z_mean, adj = self.z_mean(hidden, adj)
        z_log_std, adj = self.z_log_std(hidden, adj)
        noise = torch.randn_like(z_mean)
        z = z_mean + noise * torch.exp(z_log_std)
        logits = self.decoder(z)
        output = torch.cat([logits, z_mean, z_log_std], dim=-1)
        return output, z, adj


class ReconstructionModel(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int):
        super().__init__()
        self.graph_cnn = GraphCNN(in_channels, embedding_dim, activation='linear')
        self.decoder = LinkReconstruction()

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings, adj = self.graph_cnn(nodes, adj)
        logits = self.decoder(embeddings)
        return logits, embeddings


class ReconstructionVariationalModel(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int):
        super().__init__()
        self.z_mean = GraphCNN(in_channels, embedding_dim, activation='linear')
        self.z_log_std = GraphCNN(in_channels, embedding_dim, activation='linear')
        self.decoder = LinkReconstruction()

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_mean, adj = self.z_mean(nodes, adj)
        z_log_std, adj = self.z_log_std(nodes, adj)
        noise = torch.randn_like(z_mean)
        z = z_mean + noise * torch.exp(z_log_std)
        logits = self.decoder(z)
        output = torch.cat([logits, z_mean, z_log_std], dim=-1)
        return output, z
