############################################
# imports
############################################

import os
import shutil
import unittest
from unittest.mock import MagicMock

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from fgclustering.distance import DistanceRandomForestProximity
from fgclustering.optimizer import Optimizer
from fgclustering.clustering import ClusteringKMedoids

############################################
# Tests
############################################


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.tmp_path = os.path.join(os.getcwd(), "tmp_fgc")
        Path(self.tmp_path).mkdir(parents=True, exist_ok=True)

        self.random_state = 42
        self.n_jobs = 2
        self.verbose = 1

        self.distance_metric = DistanceRandomForestProximity(memory_efficient=False)
        self.clustering_strategy = ClusteringKMedoids(random_state=self.random_state)

        self.optimizer = Optimizer(
            distance_metric=self.distance_metric,
            clustering_strategy=self.clustering_strategy,
            random_state=self.random_state,
        )

    def tearDown(self):
        try:
            shutil.rmtree(self.tmp_path)
        except:
            pass

    def _train_model(self, model_type):
        if model_type == "cla":
            X, y = make_classification(
                n_samples=300,
                n_features=10,
                n_informative=4,
                n_redundant=2,
                n_classes=2,
                n_clusters_per_class=1,
                random_state=self.random_state,
            )
            model = RandomForestClassifier(
                max_depth=10,
                max_features="sqrt",
                max_samples=0.8,
                bootstrap=True,
                oob_score=True,
                random_state=self.random_state,
            )
        elif model_type == "reg":
            X, y = make_regression(
                n_samples=500,
                n_features=10,
                n_informative=4,
                n_targets=1,
                noise=0,
                random_state=1,
            )
            model = RandomForestRegressor(
                max_depth=5,
                max_features="sqrt",
                max_samples=0.8,
                bootstrap=True,
                oob_score=True,
                random_state=self.random_state,
            )

        X = pd.DataFrame(X)
        X.columns = [f"feature_{i}" for i in range(X.shape[1])]
        model.fit(X, y)

        return X, y, model

    def _generate_terminals(self, n_clusters, samples_per_cluster, n_trees):
        terminals = np.repeat(
            np.arange(n_trees) + np.arange(n_clusters).reshape(-1, 1) * n_trees,
            samples_per_cluster,
            axis=0,
        ).astype(np.int32)
        return terminals

    def test_optimizeK_classification(self):
        model_type = "cla"
        X, y, model = self._train_model(model_type)

        self.distance_metric.calculate_terminals(estimator=model, X=X)

        k, _, _, _ = self.optimizer.optimizeK(
            y=y,
            k_range=(2, 7),
            JI_bootstrap_iter=100,
            JI_bootstrap_sample_size=240,
            JI_discart_value=0.6,
            model_type=model_type,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        self.assertEqual(k, 6, f"Expected k=6 for classification, got {k}")

    def test_optimizeK_regression(self):
        model_type = "reg"
        X, y, model = self._train_model(model_type)

        self.distance_metric.calculate_terminals(estimator=model, X=X)

        k, _, _, _ = self.optimizer.optimizeK(
            y=y,
            k_range=(2, 7),
            JI_bootstrap_iter=100,
            JI_bootstrap_sample_size=240,
            JI_discart_value=0.8,
            model_type=model_type,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        self.assertIn(k, [2, 3, 4], f"Expected k in [2, 3, 4] for regression, got {k}")

    def test_optimizeK_output_structure(self):
        model_type = "cla"
        X, y, model = self._train_model(model_type)

        self.distance_metric.calculate_terminals(model, X)

        output = self.optimizer.optimizeK(
            y=y,
            k_range=(2, 4),
            JI_bootstrap_iter=10,
            JI_bootstrap_sample_size=200,
            JI_discart_value=0.5,
            model_type=model_type,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        self.assertEqual(len(output), 4)
        self.assertIsInstance(output[0], int)
        self.assertIsInstance(output[1], float)
        self.assertIsInstance(output[2], dict)
        self.assertIsInstance(output[3], np.ndarray)

    def test_compute_JI_stable(self):
        k = 3
        n_clusters = 3
        samples_per_cluster = 10

        self.distance_metric.terminals = self._generate_terminals(
            n_clusters=n_clusters, samples_per_cluster=samples_per_cluster, n_trees=5
        )

        cluster_labels = self.clustering_strategy.run_clustering(
            k=k,
            distance_metric=self.distance_metric,
            sample_indices=None,
            random_state_subsampling=None,
            verbose=self.verbose,
        )

        self.optimizer.n_samples_original = n_clusters * samples_per_cluster
        self.optimizer.JI_bootstrap_sample_size = 25
        self.optimizer.JI_bootstrap_iter = 100
        self.optimizer.n_jobs = self.n_jobs
        self.optimizer.verbose = self.verbose

        result = self.optimizer._compute_JI(
            k=k,
            cluster_labels_original=cluster_labels,
        )

        for v in result.values():
            self.assertEqual(v, 1.0, "Expected JI = 1.0 for perfectly stable clusters")

    def test_compute_JI_unstable(self):
        # k=2 is not aligned with actual clusters, should be less stable
        k = 2
        n_clusters = 3
        samples_per_cluster = 10

        self.distance_metric.terminals = self._generate_terminals(
            n_clusters=n_clusters, samples_per_cluster=samples_per_cluster, n_trees=5
        )

        cluster_labels = self.clustering_strategy.run_clustering(
            k=k,
            distance_metric=self.distance_metric,
            sample_indices=None,
            random_state_subsampling=None,
            verbose=self.verbose,
        )

        self.optimizer.n_samples_original = n_clusters * samples_per_cluster
        self.optimizer.JI_bootstrap_sample_size = 25
        self.optimizer.JI_bootstrap_iter = 100
        self.optimizer.n_jobs = self.n_jobs
        self.optimizer.verbose = self.verbose

        result = self.optimizer._compute_JI(
            k=k,
            cluster_labels_original=cluster_labels,
        )

        self.assertTrue(min(result.values()) < 1.0, "Expected some instability for k=2")

    def test_JI_single_bootstrap_stable(self):
        sample_size = 6
        mapping_original = {
            0: {0, 1, 2},
            1: {3, 4, 5},
        }

        # mock function which always returns the same label assignment
        mock_clustering = MagicMock()
        # this clustering should results in the same mapping as for the original clustering
        mock_clustering.run_clustering.return_value = np.array([0, 0, 0, 1, 1, 1])

        optimizer = Optimizer(
            distance_metric=self.distance_metric,
            clustering_strategy=mock_clustering,
            random_state=self.random_state,
        )
        optimizer.n_samples_original = sample_size
        optimizer.JI_bootstrap_sample_size = sample_size
        optimizer.verbose = self.verbose

        result = optimizer._compute_JI_single_bootstrap(
            k=2,
            mapping_cluster_labels_to_samples_original=mapping_original,
            random_state_subsampling=123456789,
        )

        for val in result.values():
            self.assertEqual(val, 1.0, "Expected perfect Jaccard index of 1.0 for perfect cluster match")

    def test_JI_single_bootstrap_unstable(self):
        sample_size = 6
        mapping_original = {
            0: {0, 1, 2},
            1: {3, 4, 5},
        }

        # mock function which always returns the same label assignment
        mock_clustering = MagicMock()
        # this clustering should results in a different mapping as for the original clustering
        mock_clustering.run_clustering.return_value = np.array([1, 1, 1, 1, 1, 1])

        optimizer = Optimizer(
            distance_metric=self.distance_metric,
            clustering_strategy=mock_clustering,
            random_state=self.random_state,
        )

        optimizer.n_samples_original = sample_size
        optimizer.JI_bootstrap_sample_size = sample_size
        optimizer.verbose = self.verbose

        result = optimizer._compute_JI_single_bootstrap(
            k=2,
            mapping_cluster_labels_to_samples_original=mapping_original,
            random_state_subsampling=123456789,
        )

        self.assertTrue(all(0 <= v <= 1 for v in result.values()), "Jaccard indices must be in [0, 1]")
        self.assertLess(max(result.values()), 1.0, "Expected imperfect match to result in JI < 1.0")

    def test_compute_balanced_average_impurity(self):
        # Perfect separation by class
        categorical_values = np.array([0, 0, 1, 1])
        cluster_labels = np.array([0, 0, 1, 1])

        optimizer = Optimizer(distance_metric=None, clustering_strategy=None, random_state=self.random_state)

        score = optimizer._compute_balanced_average_impurity(categorical_values, cluster_labels)

        # For perfect separation and balanced class size, Gini impurity should be 0
        self.assertAlmostEqual(score, 0.0, msg="Expected 0.0 impurity for perfect class-cluster separation")

    def test_compute_total_within_cluster_variation(self):
        # Cluster 0: variance around 1, Cluster 1: variance around 10
        continuous_values = np.array([1.0, 1.1, 0.9, 10.0, 10.1, 9.9])
        cluster_labels = np.array([0, 0, 0, 1, 1, 1])

        optimizer = Optimizer(distance_metric=None, clustering_strategy=None, random_state=self.random_state)

        score = optimizer._compute_total_within_cluster_variation(continuous_values, cluster_labels)

        # For perfect separation, variation should be around 0
        self.assertAlmostEqual(
            score, 0.0, places=3, msg="Mismatch in total within-cluster variation computation"
        )
