"""Run with python -m unittest discover -s tests -v; no pytest required."""
import copy
import importlib.util
import pickle
import unittest
from types import SimpleNamespace

import numpy as np
import torch

import fair_graphml as fg
from fair_graphml.base import NotFittedError
from fair_graphml.data import edge_groups, prepare_graph, sample_negative_edges
from fair_graphml.inprocessing.baselines import enclosing_subgraph
from fair_graphml.inprocessing.fairadj import project_simplex
from fair_graphml.inprocessing.ranking import log_sinkhorn
from fair_graphml.metrics import demographic_parity_difference, equal_opportunity_difference, ndkl

torch.set_num_threads(1)


class Graph(SimpleNamespace):
    """A minimal data carrier for testing native graph math without PyG."""
    def clone(self):
        return copy.deepcopy(self)


def toy_graph():
    generator = torch.Generator().manual_seed(7)
    pos = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                        [0, 4], [1, 5], [2, 6], [3, 7]]).t()
    neg = torch.tensor([[0, 2], [1, 3], [4, 6], [5, 7], [0, 5], [0, 6], [1, 4], [1, 6],
                        [2, 4], [2, 5], [3, 4], [3, 5]]).t()
    return Graph(x=torch.cat((torch.randn(8, 3, generator=generator), torch.zeros(8, 1)), dim=1),
                 edge_index=torch.cat((pos, pos.flip(0)), dim=1),
                 edge_label_index=torch.cat((pos, neg), dim=1),
                 edge_label=torch.cat((torch.ones(12), torch.zeros(12))),
                 sens=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
                 y=torch.tensor([2, 7, 2, 7, 2, 7, 2, 7]), num_nodes=8,
                 train_mask=torch.tensor([1, 1, 0, 0, 1, 1, 0, 0], dtype=torch.bool))


class EstimatorTests(unittest.TestCase):
    def test_all_link_estimators_train_and_predict(self):
        graph = toy_graph()
        original = graph.clone()
        for name, cls in fg.METHODS["inprocessing"].items():
            with self.subTest(method=name):
                kwargs = {"T2": 1} if name == "FairAdj" else {}
                model = cls(epochs=2, hidden_channels=8, embedding_dim=4, **kwargs)
                self.assertIs(model.fit([graph]), model)
                prob = model.predict_proba()
                self.assertEqual(prob.shape, (24, 2))
                self.assertTrue(np.isfinite(prob).all())
                np.testing.assert_allclose(prob.sum(1), 1, atol=1e-6)
                self.assertTrue(set(model.predict()).issubset({0, 1}))
                self.assertEqual(len(model.history_), 2)
                candidates = graph.edge_label_index[:, :3]
                self.assertEqual(model.predict(graph, edge_label_index=candidates).shape, (3,))
                transformed = model.transform()
                self.assertEqual(transformed.shape[0], 24 if name == "SEAL" else 8)
        torch.testing.assert_close(graph.x, original.x)
        torch.testing.assert_close(graph.edge_index, original.edge_index)
        torch.testing.assert_close(graph.edge_label, original.edge_label)

    def test_node_classifiers_use_original_class_labels(self):
        graph = toy_graph()
        for name in ("GAE", "FairGNN", "NIFTY", "FairVGNN", "DELTR", "Sinkhorn"):
            with self.subTest(method=name):
                model = getattr(fg, name)(task="node_classification", epochs=2, hidden_channels=8, embedding_dim=4).fit(graph)
                self.assertEqual(model.predict_proba().shape, (8, 2))
                self.assertTrue(set(model.predict()).issubset({2, 7}))
                self.assertEqual(model.predict(mask=graph.train_mask).shape, (4,))
                self.assertTrue(0 <= model.score() <= 1)

    def test_training_reduces_reconstruction_loss(self):
        model = fg.GAE(epochs=60, hidden_channels=16, embedding_dim=8, dropout=0, lr=0.03).fit(toy_graph())
        self.assertLess(model.history_[-1]["loss"], model.history_[0]["loss"])

    def test_fairness_coefficient_changes_training(self):
        graph = toy_graph()
        params = dict(epochs=3, dropout=0, hidden_channels=8, embedding_dim=4)
        baseline = fg.FairGNN(alpha=0, **params).fit(graph)
        constrained = fg.FairGNN(alpha=10, **params).fit(graph)
        self.assertFalse(np.allclose(baseline.predict_proba(), constrained.predict_proba()))

    def test_weights_are_nonuniform_and_positive(self):
        model = fg.UGE(epochs=1).fit(toy_graph())
        weights = model.sample_weight_[:12]
        self.assertTrue((weights > 0).all())
        self.assertGreater(float(weights.max() - weights.min()), 0)
        self.assertAlmostEqual(weights.mean().item(), 1, places=5)

    def test_group_classifiers_have_independent_parameters(self):
        model = fg.ThreeClassifiers(epochs=1).fit(toy_graph())
        self.assertEqual(len(model.model_.models), 3)
        weights = [next(v.parameters()) for v in model.model_.models]
        self.assertEqual(len({v.data_ptr() for v in weights}), 3)

    def test_holdout_node_labels_do_not_affect_training(self):
        graph = toy_graph()
        changed = graph.clone()
        changed.y[~changed.train_mask] = 1234
        a = fg.FairGNN(task="node_classification", epochs=2).fit(graph)
        b = fg.FairGNN(task="node_classification", epochs=2).fit(changed)
        np.testing.assert_array_equal(a.predict_proba(), b.predict_proba())
        np.testing.assert_array_equal(a.classes_, [2, 7])

    def test_ogb_splits_do_not_use_validation_or_test_edges(self):
        graph = toy_graph()
        pos = graph.edge_label_index[:, :12].t()
        neg = graph.edge_label_index[:, 12:].t()
        splits = {"train": {"edge": pos[:8], "edge_neg": neg[:8]},
                  "valid": {"edge": pos[8:10], "edge_neg": neg[8:10]},
                  "test": {"edge": pos[10:], "edge_neg": neg[10:]}}
        changed = copy.deepcopy(splits)
        changed["test"]["edge"] = torch.tensor([[0, 2]])
        for cls in (fg.GAE, fg.FairAdj):
            kwargs = {"T2": 1} if cls is fg.FairAdj else {}
            a = cls(epochs=2, **kwargs).fit(graph, edge_splits=splits)
            b = cls(epochs=2, **kwargs).fit(graph, edge_splits=changed)
            np.testing.assert_array_equal(a.predict_proba(), b.predict_proba())
            trained = {tuple(sorted(e)) for e in a._prepared_.edges.t().tolist()}
            self.assertFalse(any(tuple(sorted(e)) in trained for e in pos[8:].tolist()))
            # fit_transform returns the fitted training graph's representation.
            z = cls(epochs=2, **kwargs).fit_transform(graph, edge_splits=splits)
            np.testing.assert_allclose(z, a.transform())

    def test_random_state_and_repeatability(self):
        graph = toy_graph()
        torch_before = torch.random.get_rng_state().clone()
        numpy_before = np.random.get_state()
        a = fg.NIFTY(epochs=2).fit(graph)
        torch.testing.assert_close(torch.random.get_rng_state(), torch_before)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(np.random.get_state(), numpy_before)))
        b = fg.NIFTY(epochs=2).fit(graph)
        np.testing.assert_array_equal(a.transform(), b.transform())

    def test_pickle_prediction_round_trip(self):
        model = fg.FairGNN(epochs=1).fit(toy_graph())
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(model.predict_proba(), restored.predict_proba())

    def test_estimator_parameter_round_trip(self):
        for category in fg.METHODS.values():
            for name, cls in category.items():
                with self.subTest(method=name):
                    estimator = cls()
                    params = estimator.get_params(deep=False)
                    cloned = cls(**params)
                    self.assertEqual(params, cloned.get_params(deep=False))
                    self.assertFalse(cloned.__sklearn_is_fitted__())
        model = fg.FairGNN(alpha=2).set_params(alpha=3, epochs=2)
        self.assertEqual(model.get_params()["alpha"], 3)
        with self.assertRaises(ValueError):
            model.set_params(unknown=1)
        with self.assertRaises(TypeError):
            fg.FairGNN(unknown=1)

    def test_unfitted_and_invalid_data_errors(self):
        with self.assertRaises(NotFittedError):
            fg.FairGNN().predict()
        with self.assertRaises(ValueError):
            fg.GAE(epochs=0).fit(toy_graph())
        with self.assertRaises(ValueError):
            fg.GAE(epochs=1).fit([toy_graph(), toy_graph()])
        bad = toy_graph()
        bad.edge_index[0, 0] = 99
        with self.assertRaises(ValueError):
            fg.GAE(epochs=1).fit(bad)
        bad = toy_graph()
        bad.sens = torch.ones(8)
        with self.assertRaises(ValueError):
            fg.FairGNN(epochs=1).fit(bad)


class PreprocessingTests(unittest.TestCase):
    def test_all_transformers_preserve_input_and_return_fresh_graphs(self):
        graph = toy_graph()
        original = graph.clone()
        for name, cls in fg.METHODS["preprocessing"].items():
            with self.subTest(method=name):
                model = cls() if name == "FairLP" else cls(epochs=2)
                output = model.fit_transform(graph)
                self.assertEqual(output.num_nodes, 8)
                self.assertTrue(torch.isfinite(output.x).all())
                self.assertEqual(output.edge_index.shape[0], 2)
                output.x[:] = 100
                self.assertFalse(torch.all(model.transform().x == 100))
        torch.testing.assert_close(graph.x, original.x)
        torch.testing.assert_close(graph.edge_index, original.edge_index)

    def test_transformers_reject_a_different_graph(self):
        model = fg.FairLP().fit(toy_graph())
        other = toy_graph()
        other.x[0, 0] += 1
        with self.assertRaises(ValueError):
            model.transform(other)

    def test_fairlp_densities_match_up_to_edge_rounding(self):
        graph = toy_graph()
        for mode in ("avg", "min", "max"):
            result = fg.FairLP(mode=mode).fit_transform(graph)
            pairs = {tuple(sorted(e)) for e in result.edge_index.t().tolist()}
            inter = sum(graph.sens[u] != graph.sens[v] for u, v in pairs)
            intra = len(pairs) - inter
            self.assertLessEqual(abs(inter / 16 - intra / 12), 1 / 12 + 1 / 16)

    def test_rewiring_removes_stale_edge_fields(self):
        graph = toy_graph()
        graph.edge_attr = torch.ones(graph.edge_index.shape[1], 2)
        graph.adj_t = torch.eye(8)
        result = fg.FairLP().fit_transform(graph)
        self.assertIsNone(result.edge_attr)
        self.assertIsNone(result.adj_t)
        self.assertIsNotNone(graph.edge_attr)

    def test_edits_feature_scales_and_adjacency_constraints(self):
        model = fg.EDITS(epochs=2).fit(toy_graph())
        self.assertTrue(np.all(model.feature_scale_ >= 0))
        self.assertTrue(np.all(model.feature_scale_ <= 1))
        self.assertTrue(np.all(model.learned_adjacency_ >= 0))
        np.testing.assert_allclose(model.learned_adjacency_, model.learned_adjacency_.T)
        self.assertTrue(torch.isfinite(model.transform().x).all())

    def test_graphair_has_learned_augmentation_parameters(self):
        graph = toy_graph()
        model = fg.Graphair(epochs=2, embedding_dim=4).fit(graph)
        self.assertEqual(model.get_embeddings().shape, (8, 4))
        self.assertTrue(np.isfinite(model.learned_adjacency_).all())
        self.assertGreater(sum(p.numel() for p in model.model_["augmenter"].parameters()), 0)
        np.testing.assert_allclose(model.learned_adjacency_, model.learned_adjacency_.T)


class RankingAndEmbeddingTests(unittest.TestCase):
    def test_detconstsort_prefix_quotas_and_relevance(self):
        scores = np.arange(12, 0, -1)
        groups = np.array([0] * 8 + [1] * 4)
        ranking = fg.DetConstSort(k=8, distribution={0: 0.5, 1: 0.5}).fit_transform(scores, sensitive_features=groups)
        self.assertEqual(len(np.unique(ranking)), 8)
        for prefix in range(1, 9):
            self.assertGreaterEqual(np.sum(groups[ranking[:prefix]] == 0), prefix // 2)
            self.assertGreaterEqual(np.sum(groups[ranking[:prefix]] == 1), prefix // 2)
        self.assertLess(ndkl(ranking, groups, {0: 0.5, 1: 0.5}), ndkl(np.arange(8), groups, {0: 0.5, 1: 0.5}))
        with self.assertRaises(ValueError):
            fg.DetConstSort(k=12, distribution={0: 0.5, 1: 0.5}).fit(scores, sensitive_features=groups)

    def test_rerank_extreme_distributions_terminate(self):
        for distribution, k in (({0: 0.0, 1: 1.0}, 2), ({0: 1e-12, 1: 1 - 1e-12}, 3)):
            ranking = fg.DetConstSort(distribution=distribution, k=k).fit_transform([1, 2, 3], sensitive_features=[0, 1, 1])
            np.testing.assert_array_equal(ranking, [2, 1, 0][:k])

    def test_epsilon_greedy_seed_and_score_alignment(self):
        scores, groups = np.arange(10), np.arange(10) % 2
        np.testing.assert_array_equal(fg.EpsilonGreedy(epsilon=0).fit_transform(scores, sensitive_features=groups), np.arange(9, -1, -1))
        a = fg.EpsilonGreedy(epsilon=1, random_state=8).fit(scores, sensitive_features=groups)
        b = fg.EpsilonGreedy(epsilon=1, random_state=8).fit(scores, sensitive_features=groups)
        np.testing.assert_array_equal(a.transform(), b.transform())
        np.testing.assert_array_equal(a.ranked_scores_, scores[a.ranking_])

    def test_pyg_candidate_style_reranking(self):
        graph = toy_graph()
        result = fg.DetConstSort(k=10).fit_transform(graph, scores=np.linspace(0, 1, 24))
        self.assertEqual(result.shape, (10,))

    def test_walk_training_dimensions_isolates_and_repeatability(self):
        graph = toy_graph()
        graph.edge_index = graph.edge_index[:, (graph.edge_index != 7).all(0)]
        params = dict(embedding_dim=4, epochs=1, walk_length=4, num_walks=1, context_size=2)
        for cls in (fg.FairWalk, fg.Node2Vec):
            a, b = cls(**params).fit(graph), cls(**params).fit(graph)
            self.assertEqual(a.transform().shape, (8, 4))
            self.assertTrue(np.isfinite(a.transform()).all())
            np.testing.assert_array_equal(a.transform(), b.transform())
            self.assertEqual(a.transition_probabilities(7), {})

    def test_fairwalk_balances_groups_in_transition_probabilities(self):
        graph = toy_graph()
        graph.edge_index = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]])
        kwargs = dict(embedding_dim=2, epochs=1, walk_length=2, num_walks=1, context_size=1)
        fair = fg.FairWalk(**kwargs).fit(graph).transition_probabilities(0)
        biased = fg.Node2Vec(**kwargs).fit(graph).transition_probabilities(0)
        self.assertAlmostEqual(sum(fair[v] for v in (1, 2, 3)), 0.5)
        self.assertAlmostEqual(fair[4], 0.5)
        self.assertAlmostEqual(biased[4], 0.25)


class MathAndDataTests(unittest.TestCase):
    def test_structural_heuristics_match_known_graph(self):
        graph = toy_graph()
        graph.edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
        candidates = torch.tensor([[0, 0, 1], [3, 1, 2]])
        common = fg.CommonNeighbors().fit(graph)
        np.testing.assert_array_equal(common.predict(edge_label_index=candidates), [2, 0, 2])
        aa = fg.AdamicAdar().fit(graph).predict(edge_label_index=candidates)
        np.testing.assert_allclose(aa, [2 / np.log(2), 0, 2 / np.log(2)])
        ppr = fg.PersonalizedPageRank().fit(graph)
        first = ppr.predict(edge_label_index=candidates)
        np.testing.assert_array_equal(first, ppr.predict(edge_label_index=candidates))
        self.assertTrue(np.all((first >= 0) & (first <= 1)))

    def test_negative_sampling_excludes_reverse_edges_and_loops(self):
        graph = toy_graph()
        negatives = sample_negative_edges(graph.edge_index, 8, 100)
        occupied = {tuple(sorted(e)) for e in graph.edge_index.t().tolist()}
        pairs = [tuple(sorted(e)) for e in negatives.t().tolist()]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertFalse(set(pairs) & occupied)
        self.assertTrue(all(u != v for u, v in pairs))

    def test_pyg_randomlinksplit_fields_are_respected(self):
        graph = toy_graph()
        p = prepare_graph(graph)
        torch.testing.assert_close(p.target_edges, graph.edge_label_index)
        torch.testing.assert_close(p.targets, graph.edge_label)

    def test_sensitive_labels_are_recoded_and_edge_groups_are_symmetric(self):
        graph = toy_graph()
        graph.sens = graph.sens * 10 + 3
        p = prepare_graph(graph, require_sensitive=True)
        torch.testing.assert_close(p.groups, torch.tensor([3, 13]))
        torch.testing.assert_close(edge_groups(p.sens, p.edges), edge_groups(p.sens, p.edges.flip(0)))

    def test_simplex_projection_and_sinkhorn_normalization(self):
        rows = torch.tensor([[-1.0, 2.0, 0.5], [5.0, -3.0, 4.0]])
        result = project_simplex(rows)
        self.assertTrue((result >= 0).all())
        torch.testing.assert_close(result.sum(1), torch.ones(2))
        matrix = log_sinkhorn(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 30)
        torch.testing.assert_close(matrix.sum(0), torch.ones(2))
        torch.testing.assert_close(matrix.sum(1), torch.ones(2))

    def test_seal_removes_target_edge_and_labels_centers(self):
        graph = toy_graph()
        x, a, labels = enclosing_subgraph(graph.x, graph.edge_index, [0, 1])
        self.assertEqual(labels[:2].tolist(), [1, 1])
        self.assertEqual(a.to_dense()[0, 1].item(), 0)
        self.assertGreaterEqual(len(x), 2)

    def test_fairness_metrics(self):
        pred = [1, 1, 0, 1]
        sens = [0, 0, 1, 1]
        self.assertEqual(demographic_parity_difference(pred, sens), 0.5)
        self.assertEqual(equal_opportunity_difference([1, 0, 1, 1], pred, sens), 0.5)
        self.assertTrue(np.isnan(equal_opportunity_difference([1, 1, 0, 0], pred, sens)))
        self.assertEqual(ndkl([0, 1], [0, 0]), 0)


@unittest.skipUnless(importlib.util.find_spec("torch_geometric"), "Optional PyG dependency is not installed")
class PyGIntegrationTests(unittest.TestCase):
    def test_real_data_random_link_split(self):
        from torch_geometric.data import Data
        from torch_geometric.transforms import RandomLinkSplit
        toy = toy_graph()
        data = Data(x=toy.x, edge_index=toy.edge_index, y=toy.y, sens=toy.sens, train_mask=toy.train_mask)
        train, val, test = RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=True)(data)
        model = fg.FairGNN(epochs=2).fit(train)
        self.assertEqual(model.predict_proba(test).shape, (test.edge_label.numel(), 2))
        self.assertEqual(fg.FairLP().fit_transform(train).num_nodes, 8)

    def test_pyg_inmemory_dataset(self):
        from torch_geometric.data import Data, InMemoryDataset
        toy = toy_graph()
        data = Data(x=toy.x, edge_index=toy.edge_index, sens=toy.sens)
        class Dataset(InMemoryDataset):
            def __init__(self):
                super().__init__(root=None)
                self.data, self.slices = self.collate([data])
        model = fg.GAE(epochs=1).fit(Dataset())
        self.assertEqual(model.transform().shape, (8, 16))

    def test_native_gcn_matches_pyg_gcn(self):
        from torch_geometric.nn import GCNConv
        from fair_graphml.nn import adjacency, normalize
        graph = toy_graph()
        conv = GCNConv(4, 3)
        a = normalize(adjacency(graph.edge_index, 8, self_loops=True))
        expected = a @ graph.x @ conv.lin.weight.t() + conv.bias
        torch.testing.assert_close(conv(graph.x, graph.edge_index), expected)


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Optional sklearn dependency is not installed")
class SklearnIntegrationTests(unittest.TestCase):
    def test_clone_and_pipeline(self):
        from sklearn.base import clone
        from sklearn.pipeline import Pipeline
        graph = toy_graph()
        for methods in fg.METHODS.values():
            for cls in methods.values():
                original = cls()
                self.assertEqual(clone(original).get_params(), original.get_params())
        pipeline = Pipeline([("pre", fg.FairLP()), ("model", fg.FairGNN(epochs=2))])
        pipeline.fit(graph)
        self.assertEqual(pipeline.predict_proba(graph).shape, (24, 2))


if __name__ == "__main__":
    unittest.main()
