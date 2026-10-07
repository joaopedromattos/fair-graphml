# Method provenance and porting decisions

The library consolidates the primary methods in the four experiment folders,
plus the GAE, neural common-neighbor, and SEAL predictors used with them.
It also exposes common neighbors, Adamic–Adar, and personalized PageRank from
`pre_processing_methods/structural_heuristics/common_neighbors.ipynb` and the
`CN`, `AA`, and `PPR` functions in `pre_processing_methods/FairLP/heuristics.py`.
These estimators return raw structural link scores. Natural logarithms are
the Adamic–Adar default; `log_base=10` matches the notebook's variant.
Dataset loaders, plotting scripts, hyperparameter searches, duplicate model
definitions, and nested external baseline projects are not public estimators.
The original files remain in place. The package does not import files outside
`fair_graphml`.

## In-processing

| Class | Repository source | Implementation and changes |
| --- | --- | --- |
| `FairGNN` | `in_processing_methods/FairGNN.py` | Sensitive-label estimator, covariance penalty, and alternating adversary. Adds explicit sensitive-estimator supervision, which the source's detached target prevented. Uses the absolute cross-covariance rather than the source link loop's mean absolute per-sample product. Categorical sensitive labels generalize the binary estimator. |
| `NIFTY` | `in_processing_methods/NIFTY.py` | Spectrally normalized projection and prediction heads, stop-gradient cosine similarity, two alternating optimizer steps, and supervised losses on both views. Retains Gaussian feature perturbation and optional binary sensitive-feature flips. Configured augmentation rates are honored; the source overwrote training rates with zero. |
| `FairVGNN` | `in_processing_methods/FairVGNN.py` | Gumbel-softmax channel masks; discriminator, classifier, and generator updates; uniform adversarial confusion; mask regularization and channel contribution clipping. Uses one mask sample per step and exposes step counts. Supports categorical groups through a softmax discriminator. |
| `DELTR` | `in_processing_methods/DELTR.py` | Stable softmax exposure per group member and one-sided squared exposure gap. Uses a specified protected group versus remaining groups. The source's pairwise ranking-loss option and multiple exposure variants are not exposed. |
| `UGE` | `in_processing_methods/UGE.py` | Weighted link BCE, analytic sensitive-pair importance weights, and optional group-rate regularization. The source's active PyG path created all-one weights and did not apply its debiasing switches. `uge-w` activates importance weighting; `uge-r` and `uge-c` use a group-rate penalty extension, not a reproduction of the paper's embedding regularizer. Uses caller-supplied graph splits rather than resplitting internally. |
| `FairAdj` / `FairADJ` | `in_processing_methods/FairAdj/FairAdj.py`, `model/optimizer.py` | Alternating variational encoder training and simplex-projected adjacency fairness updates over intra/inter node pairs. Corrects the original training loop's use of test edges. Uses stable feature handling and an explicit dense-memory guard. |
| `Sinkhorn` | `in_processing_methods/sinkhorn.py` | Ports its **active** score-weighted soft-label prefix-KL loss: softmax of score times group one-hot labels, score ordering, discounted reverse KL, and supervised BCE. The source does not call its experimental Sinkhorn permutation helper in this fit path; this class does not pretend that it does. The mathematical `log_sinkhorn` helper is separately available in `inprocessing.ranking`. |
| `ThreeClassifiers` | `in_processing_methods/three_classifiers.py` | Independent encoder/decoder per unordered sensitive-group pair, with a mean of group-specific BCE losses. Binary sensitive labels yield three models. A full-batch optimizer replaces three loader/optimizer loops; parameters remain independent. Every group must have training candidates. Prediction retains fitted sensitive node IDs. |
| `SinkhornMultiClass` / `ThreeClassifiersMultiClass` | `in_processing_methods/three_classifiers_multiclass.py` | The file's active implementation is a **single binary link classifier trained with BCE**. There is no multiclass Sinkhorn loss in that implementation. This class exposes that behavior. |
| `GAE` | `in_processing_methods/GAE.py` | Two-layer graph encoder, inner-product decoder, edge-candidate BCE. Uses sparse candidate decoding rather than materializing an N-by-N reconstruction. |
| `NCN` | `in_processing_methods/NCNC.py`, `CNLinkPredictor` | Endpoint embedding products plus the sum of common-neighbor embeddings, mapped through MLPs and a learned beta. Replaces compiled sparse intersections with Python sets. Simplifies internal MLP depth and omits degree sampling, incomplete-neighbor completion variants, and optional multi-domain dropout. |
| `SEAL` | `in_processing_methods/SEAL.py`, `SEALDataset`, `GCN_SEAL` | Extracts enclosing subgraphs, removes the target edge, applies double-radius node labels, and uses three graph-convolution layers and center-product pooling. Extracts per candidate without disk caches or a compiled loader. `transform` returns edge logits; SEAL has no global node embedding. |

Supervised ports share a two-layer encoder with optional GCN, GraphSAGE, GIN,
or MLP propagation. Architectures, scalar binary output heads, initializers,
full-batch schedules, and common hyperparameter defaults differ from individual
experiments. They should not be used to claim reproduction of paper metrics
without separate benchmark validation. There is no implicit validation-based
early stopping; all requested epochs run and `history_` records their losses.

Node classification is available on the shared supervised classes as a
consistent extension. UGE, FairAdj, ThreeClassifiers, NCN, and SEAL retain
link-only tasks and fail explicitly on unsupported node-classification requests.

## Pre-processing

| Class | Repository source | Implementation and changes |
| --- | --- | --- |
| `FairLP` | `pre_processing_methods/FairLP/fairlp.py` | Algorithm copied into `_vendor/fairlp.py`. Removes progress printing, adds graph cloning and PyG conversion, and discards self links when forming the simple graph. The protected group is explicitly chosen from original numeric group labels. |
| `EDITS` | `pre_processing_methods/EDITS/EDITS.py` | Copies the optimizer, proximal operators, feature scale module, Wasserstein critic, propagation, and adjacency update core into `_vendor/edits.py`. Removes fixed CUDA/half-precision use and per-epoch reseeding, fixes optimizer deserialization, stabilizes constant feature normalization, and adds a graph-transformer lifecycle. Exports scaled features in their original units; optional truncation reproduces feature removal. Adjacency export uses an explicit threshold rather than the experiment's external binarization stage. |
| `EDITSBatched` | `pre_processing_methods/EDITS/EDITS_batched.py` | Batches feature optimization and uses full-graph structural updates. Stratifies batches to prevent a critic update with an empty sensitive group. Does not remove the dense adjacency memory cost. |
| `Graphair` / `GraphAir` | `pre_processing_methods/fairgraph/method/Graphair/{graphair,aug_module,GCN}.py` | Learned edge/feature augmentation with straight-through relaxed Bernoulli sampling, InfoNCE, sensitive adversary, edge reconstruction on training candidates, and feature reconstruction. Native torch sampling replaces Pyro. Uses a shared encoder, includes projector parameters in the optimizer, and exports deterministic probability-based augmentations. No GraphSAINT batch sampler or disk checkpoint side effects. |
| `FairEGM` | `pre_processing_methods/FairEGM/fair_egm.py`, `torch_layers/`, `torch_models/` | Copies PyTorch layers and losses into `_vendor/fairegm`. Adapts DenseGraphConv to native torch algebra: `lin_rel(A @ X) + lin_root(X)`. Retains alternating task/fair optimizers and exposes one-hot categorical sensitive attributes. Produces a representation graph with fair embeddings in `x`; the internal fair adjacency is a separate learned attribute. |
| `GFO`, `CFO`, `FEW` | `pre_processing_methods/FairEGM/torch_layers/{gfo_graph_conv,cfo_graph_conv,few_graph_conv}.py` | Convenience classes fixing FairEGM's layer choice. Layer formulas and constraint mechanisms are retained. |

These graph transformations are transductive. They preserve the caller's graph
and return a fresh output. Candidate labels/masks are copied, while edge-aligned
features and cached adjacency fields are invalidated when topology changes.
Fit them on the message-passing training structure before evaluating links.

## Embeddings and post-processing

| Class | Repository source | Implementation and changes |
| --- | --- | --- |
| `FairWalk` | `node_embeddings_methods/FairWalk.py` | Activates group-balanced walk transitions from the source's probability/walk machinery. Its active experiment fit path had bypassed these walks and hardcoded an ordinary Node2Vec model and dimension. Uses native torch negative-sampling skip-gram, honors dimensions and p/q, and preserves isolated nodes. |
| `Node2Vec` | `node_embeddings_methods/FairWalk.py`, `Node2Vec` | Ordinary second-order transitions and native torch skip-gram, using symmetric context windows. Removes the compiled random-walk requirement. |
| `DetConstSort` / `DETCONSTSORT` | `pos_processing_methods/methods.py`, `DETCONSTSORT` | Removes the pandas-only API and printing; preserves score-ordered insertion with prefix deadlines. Checks quotas, handles zero target mass, skips empty deadline ranges, and raises on infeasible constraints. Returns indices so score identity is preserved. |
| `EpsilonGreedy` / `EPSILONGREEDY` | `pos_processing_methods/methods.py`, `EPSILONGREEDY` | Preserves seeded random swaps, using a local NumPy RandomState. Initially sorts candidate scores by relevance. |

`fair_graphml.metrics` contains dependency-light versions of group positive-rate
gaps, equal-opportunity gaps, and evaluation NDKL. Evaluation NDKL uses
KL(prefix || target); the repo's differentiable training loss uses the reverse
KL direction. Neither metric silently replaces an undefined rate with zero.

The sources include material from several research implementations. This
package does not invent a common upstream license or claim ownership of copied
algorithms; retain and review the original projects' attribution and license
terms before redistributing it externally.
