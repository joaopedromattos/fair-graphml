"""Offline end-to-end demo. Uses real PyG Data when PyG is installed."""
from copy import deepcopy
from types import SimpleNamespace

import torch

from fair_graphml import FairLP, FairGNN, FairWalk, DetConstSort, list_methods

try:
    from torch_geometric.data import Data
except ImportError:
    class Data(SimpleNamespace):
        def clone(self):
            return deepcopy(self)
    print("PyG is not installed; this demo uses a lightweight graph data carrier.")


def main():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(10)
    positive = torch.tensor([[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [0, 4], [1, 5]]).t()
    negative = torch.tensor([[0, 2], [1, 3], [4, 6], [5, 7], [0, 5], [1, 4], [2, 7], [3, 6]]).t()
    data = Data(x=torch.randn(8, 5, generator=generator),
                edge_index=torch.cat((positive, positive.flip(0)), dim=1),
                edge_label_index=torch.cat((positive, negative), dim=1),
                edge_label=torch.cat((torch.ones(8), torch.zeros(8))),
                sens=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]), num_nodes=8)

    fair_data = FairLP().fit_transform(data)
    model = FairGNN(epochs=20, hidden_channels=16, embedding_dim=8).fit(fair_data)
    scores = model.predict_proba()[:, 1]
    ranking = DetConstSort(k=8).fit_transform(data, scores=scores)
    embeddings = FairWalk(embedding_dim=8, epochs=1, walk_length=5, num_walks=1).fit_transform(data)
    print("Methods:", list_methods())
    print("Link probabilities:", scores.round(3))
    print("Fair ranking (candidate indices):", ranking)
    print("FairWalk embedding shape:", embeddings.shape)


if __name__ == "__main__":
    main()
