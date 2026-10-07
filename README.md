# fair-graphml

Graph fairness methods from this repository, collected into a standalone Python
package with an estimator interface. Instantiate a class, call `fit`, then use
`predict`, `predict_proba`, or `transform`.

The original experiment folders are unchanged. The package runs independently
of them: it contains copied components and native PyTorch ports, with no
`sys.path` edits, dataset downloads at import time, experiment tracking, or
automatic checkpoint writes.

## Install

Clone the library and install it from its root:

```bash
git clone git@github.com:joaopedromattos/fair-graphml.git
cd fair-graphml
pip install -e '.[pyg,sklearn]'
```

From the original `fairness` workspace, use
`pip install -e './fair-graphml[pyg,sklearn]'` instead.

The distribution is named **fair-graphml**; the Python import is `fair_graphml`.
PyG and scikit-learn are optional integrations. The core requires PyTorch,
NumPy, SciPy, and NetworkX and uses native torch graph operations, so compiled
`torch-scatter`, `torch-sparse`, `torch-cluster`, and `pyg-lib` are unnecessary.
Install the core alone with `pip install -e .` from the library root.

The library also supports the repository's `ss` environment (Python 3.9,
PyTorch 1.13.1, PyG 2.3.0, and scikit-learn 1.2.2); the declared minimum
versions accommodate this stack without requiring framework upgrades.
NetworkX is a required dependency and is not currently installed in `ss`.
All 35 tests, including the real PyG and sklearn integrations, passed using
`ss` with an existing local NetworkX 2.8.7 copy on a temporary dependency path.
The environment itself was not modified.

## Train on a PyG dataset

```python
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from fair_graphml import FairGNN

dataset = Planetoid(root="data/Planetoid", name="Cora")
data = dataset[0]
data.sens = sensitive_attributes  # your numeric categorical group per node

train, valid, test = RandomLinkSplit(
    num_val=0.1, num_test=0.2, is_undirected=True,
    add_negative_train_samples=True,
)(data)

model = FairGNN(epochs=100, hidden_channels=64, embedding_dim=32, alpha=4.0)
model.fit(train)
probabilities = model.predict_proba(test)[:, 1]
predictions = model.predict(test)
embeddings = model.transform()  # [num_nodes, embedding_dim]
```

`sensitive_attributes` is supplied by your application; ordinary Planetoid
datasets do not define a demographic sensitive attribute. Do not use the
prediction target as a sensitive attribute unless that is your intended task.
`Data` objects and datasets containing one graph are accepted. For a dataset
with multiple graphs, pass `dataset[i]`; these methods train on one graph at a
time, rather than performing graph-level classification.

For node classification, change the task:

```python
model = FairGNN(task="node_classification", epochs=100)
model.fit(dataset, sensitive_features=sensitive_attributes)
predictions = model.predict(mask=data.test_mask)
probabilities = model.predict_proba(mask=data.test_mask)
```

Node classification uses `data.y` and `data.train_mask`. `y=...`, `train_mask=...`,
and `sensitive_train_mask=...` can be passed explicitly to `fit`. Only training
node labels establish the class vocabulary; holdout labels are unused for
optimization. Multiclass targets and noncontiguous numeric sensitive labels are
supported by the shared classifiers.

## Methods

| Category | Classes |
| --- | --- |
| In-processing | `FairGNN`, `FairVGNN`, `NIFTY`, `UGE`, `DELTR`, `FairAdj`, `Sinkhorn`, `ThreeClassifiers`, `SinkhornMultiClass` |
| Pre-processing | `FairLP`, `EDITS`, `EDITSBatched`, `Graphair`, `FairEGM`, `GFO`, `CFO`, `FEW` |
| Node embeddings | `FairWalk`, `Node2Vec` |
| Post-processing | `DetConstSort`, `EpsilonGreedy` |
| Link prediction baselines | `GAE`, `NCN`, `SEAL` |
| Structural link scores | `CommonNeighbors`, `AdamicAdar`, `PersonalizedPageRank` |

All are available from `fair_graphml` and their category subpackages:
`fair_graphml.inprocessing`, `.preprocessing`, `.embedding`, `.postprocessing`.
Structural scores are also available from `fair_graphml.heuristics`; their
`predict` returns raw link scores without training a classifier.
`FairADJ`, `GraphAir`, and `ThreeClassifiersMultiClass` are compatibility aliases.
Use `fair_graphml.list_methods()` to discover the registered methods.

These are **API-oriented ports**, not a claim of bitwise reproduction of the
original experiments. Shared encoders and training infrastructure replace the
experiment-specific architectures. Some existing defects are corrected and
some source variants have surprising names. The [method provenance and
adaptation notes](docs/METHODS.md) identify each implementation and its changes.

## Preprocess, train, and rerank

```python
import numpy as np
from fair_graphml import FairLP, NIFTY, DetConstSort

preprocessor = FairLP(mode="avg", protected_group=1)
fair_train = preprocessor.fit_transform(train)

model = NIFTY(epochs=100, sim_coeff=0.5)
model.fit(fair_train)

# Use the fair training structure when encoding held-out candidates.
scores = model.predict_proba(edge_label_index=test.edge_label_index)[:, 1]

reranker = DetConstSort(k=min(100, len(scores)))
ranking = reranker.fit_transform(
    test, scores=scores, edge_label_index=test.edge_label_index,
)
ranked_candidate_edges = test.edge_label_index[:, ranking]
ranked_scores = scores[ranking]
```

Preprocessors return fresh `Data` objects and preserve node IDs, labels, masks,
and candidate edges. Rewiring invalidates stale `edge_attr`, `adj`, and `adj_t`.
They learn edits for one graph and reject a different graph at `transform` time.
Fit preprocessing on the **training graph**. If validation/test `Data` objects
carry another message-passing graph, explicitly supply held-out candidate edges
to the model fitted on the transformed training graph, as above.

`FairEGM.fit_transform` places fair node embeddings in the output graph's `x`.
`Graphair.fit_transform` returns learned graph augmentations, and
`Graphair.get_embeddings()` returns its contrastive representations. EDITS
exposes `feature_scale_` and `learned_adjacency_`; FairLP exposes `added_edges_`
and `removed_edges_`.

For array-based reranking, provide **one group per candidate**, not per node:

```python
ranking = DetConstSort(k=4, distribution={0: 0.5, 1: 0.5}).fit_transform(
    np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
    sensitive_features=np.array([0, 0, 0, 1, 1, 1]),
)
```

Rerankers return candidate **indices**, preserving the original scores. Their
`ranking_` and `ranked_scores_` attributes are available after fitting.
`distribution=None` uses the observed candidate group distribution. Infeasible
prefix quotas raise `ValueError`. EpsilonGreedy performs seeded random swaps;
it offers no prefix fairness guarantee.

## Node embeddings

```python
from fair_graphml import FairWalk
from sklearn.linear_model import LogisticRegression

embedder = FairWalk(embedding_dim=64, epochs=5, random_state=42)
z = embedder.fit_transform(train)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(z[data.train_mask.cpu().numpy()], data.y[data.train_mask].cpu().numpy())
predictions = classifier.predict(z[data.test_mask.cpu().numpy()])
```

FairWalk balances the probability of visiting each **available neighbor group**.
Node2Vec uses the ordinary second-order transition rule. Both train native
PyTorch skip-gram embeddings, honor `p`, `q`, and `embedding_dim`, and retain
isolated nodes in their output. Embeddings are transductive: node indices keep
their fitted meaning.

## Existing repository edge splits

```python
import torch

model = FairGNN(epochs=100)
model.fit(data, sensitive_features=sens, edge_splits=splits)
test_edges = torch.cat((splits["test"]["edge"], splits["test"]["edge_neg"])).t()
scores = model.predict_proba(edge_label_index=test_edges)[:, 1]
```

This format expects `splits["train"]["edge"]` and `edge_neg` in `[E, 2]`
shape. The adapter rebuilds the message-passing graph from training positives;
it does not use validation or test edges or labels. PyG-style
`edge_label_index` uses `[2, E]` shape and binary `edge_label`/`y` values.
Without explicit link candidates, the adapter uses the supplied graph's
positive edges and samples distinct non-edges, excluding reverse edges for
undirected graphs. Supply already-split graphs when evaluating held-out links.

## Common estimator options

Supervised classes accept `epochs`, `lr`, `hidden_channels`, `embedding_dim`,
`dropout`, `weight_decay`, `device`, `random_state`, `encoder`, and `decoder`.
Available shared encoders are `gcn`, `sage`, `gin`, and `mlp`; shared decoders
are `dot` and `mlp`. NCN and SEAL use their own decoders. CPU is the default;
pass `device="cuda:0"` explicitly to use a GPU.

Method parameters are direct constructor keywords, for example
`FairGNN(alpha=4, beta=0.01)`, `DELTR(exposure_coeff=1)`, or
`FairEGM(mode="few")`. Unknown keywords fail immediately. `get_params`,
`set_params`, sklearn `clone`, and sklearn `Pipeline` are supported. A graph
is one estimator input; ordinary row-based sklearn cross-validation does not
split graph nodes or edges. Build graph-aware splits explicitly.

```python
from sklearn.pipeline import Pipeline
from fair_graphml import FairLP, FairGNN

pipeline = Pipeline([("pre", FairLP()), ("model", FairGNN(epochs=100))])
pipeline.fit(train)  # train.sens supplies sensitive attributes to both steps
predictions = pipeline.predict(train)
```

`fit` returns `self`. `predict` returns labels in the original class vocabulary.
`predict_proba` returns `[E, 2]` for links and `[N, C]` for nodes.
`decision_function` returns logits. `transform` returns node embeddings for
the shared models. SEAL learns enclosing-subgraph representations and returns
one edge logit per candidate from `transform`, rather than global node embeddings.
`history_` contains per-epoch loss components. Random seeds are local to fitting;
the input graph and global NumPy/PyTorch random states are preserved.

FairLP enumerates potential edges. EDITS, FairAdj, Graphair, and FairEGM use
dense adjacency matrices. Their default node-count guards prevent accidental
quadratic allocations; larger graphs require an explicit guard increase.
EDITSBatched batches feature updates but still stores dense adjacency. These
ports do not implement neighbor-sampled training for large graphs. NCN and SEAL
use correctness-first Python graph traversal and are intended for small graphs.

Fairness metrics are available in `fair_graphml.metrics`:
`demographic_parity_difference`, `equal_opportunity_difference`, and `ndkl`.
Undefined group true-positive rates produce `NaN` rather than a fabricated gap.

## Examples and validation

Run from the library root:

```bash
PYTHONPATH=src python examples/quickstart.py
PYTHONPATH=src python -m unittest discover -s tests -v
python -m pip wheel --no-deps --no-build-isolation .
```

The test suite trains every registered estimator on a small graph and checks
split leakage, input preservation, parameter cloning, deterministic fitting,
fairness coefficients, group-balanced walk transitions, ranking quotas, and
graph constraints. Real PyG and sklearn integration tests run when those
optional dependencies are installed, and report skips otherwise.
