import torch
from torch import nn


class LinkReconstruction(nn.Module):
    """Decoder that reconstructs adjacency scores from embeddings."""

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        return torch.matmul(embeddings, embeddings.transpose(-1, -2))
