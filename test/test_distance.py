############################################
# imports
############################################

import os
import shutil
import unittest
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from fgclustering.distance import (
    DistanceRandomForestProximity,
    DistanceWasserstein,
    DistanceJensenShannon,
)

############################################
# Tests
############################################


class TestDistanceRandomForestProximity(unittest.TestCase):
    def setUp(self):
        self.tmp_path = os.path.join(os.getcwd(), "tmp_fgc")
        Path(self.tmp_path).mkdir(parents=True, exist_ok=True)

        self.random_state = 42
        self.n_jobs = 2
        self.verbose = 1

        self.X, self.y, self.model = self._train_model()

    def _train_model(self):

        # Generate test data
        X, y = make_classification(
            n_samples=50,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            random_state=self.random_state,
        )
        X = pd.DataFrame(data=X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y = y

        model = RandomForestClassifier(
            max_depth=10,
            max_features="sqrt",
            max_samples=0.8,
            bootstrap=True,
            oob_score=True,
            random_state=self.random_state,
        )
        model.fit(X=X, y=y)

        return X, y, model

    def tearDown(self):
        try:
            shutil.rmtree(self.tmp_path)
        except OSError:
            pass

    def test_calculate_terminals(self):
        dist = DistanceRandomForestProximity()
        dist.calculate_terminals(estimator=self.model, X=self.X)
        self.assertIsNotNone(obj=dist.terminals)
        self.assertEqual(first=dist.terminals.shape[0], second=self.X.shape[0])

    def test_calculate_distance_matrix_non_memory_efficient(self):
        dist = DistanceRandomForestProximity(memory_efficient=False)
        dist.calculate_terminals(estimator=self.model, X=self.X)
        matrix, file = dist.calculate_distance_matrix(sample_indices=None)
        self.assertEqual(first=matrix.shape[0], second=matrix.shape[1])
        self.assertEqual(first=matrix.shape[0], second=len(self.X))
        self.assertTrue(expr=np.allclose(a=matrix, b=matrix.T))
        self.assertTrue(expr=np.all(a=np.diag(v=matrix) == 0))
        self.assertTrue(expr=file is None)

    def test_calculate_distance_matrix_memory_efficient(self):
        dist = DistanceRandomForestProximity(
            memory_efficient=True, dir_distance_matrix=self.tmp_path
        )
        dist.calculate_terminals(estimator=self.model, X=self.X)
        matrix, file = dist.calculate_distance_matrix(sample_indices=None)
        self.assertTrue(isinstance(matrix, np.memmap))
        self.assertEqual(first=matrix.shape[0], second=matrix.shape[1])
        self.assertTrue(expr=np.allclose(a=matrix, b=matrix.T))
        self.assertTrue(expr=np.all(a=np.diag(v=matrix) == 0))
        self.assertTrue(expr=os.path.exists(path=file))

    def test_calculate_distance_matrix_error_without_terminals(self):
        dist = DistanceRandomForestProximity()
        with self.assertRaises(ValueError):
            _ = dist.calculate_distance_matrix(sample_indices=None)

    def test_calculate_distance_matrix_error_missing_dir_in_memory_efficient_mode(self):
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(memory_efficient=True)

    def test_calculate_distance_matrix_with_sample_indices(self):
        dist = DistanceRandomForestProximity(memory_efficient=False)
        dist.calculate_terminals(estimator=self.model, X=self.X)
        sample_indices = np.random.choice(a=len(self.X), size=20, replace=False)
        matrix, file = dist.calculate_distance_matrix(sample_indices=sample_indices)
        self.assertEqual(first=matrix.shape, second=(20, 20))

    def test_min_samples_in_node_none_matches_baseline(self):
        """Default behavior (min_samples_in_node=None) must match pre-feature output."""
        dist_baseline = DistanceRandomForestProximity()
        dist_baseline.calculate_terminals(estimator=self.model, X=self.X)
        baseline_matrix, _ = dist_baseline.calculate_distance_matrix(
            sample_indices=None
        )

        dist_new = DistanceRandomForestProximity(min_samples_in_node=None)
        dist_new.calculate_terminals(estimator=self.model, X=self.X)
        new_matrix, _ = dist_new.calculate_distance_matrix(sample_indices=None)

        np.testing.assert_array_equal(baseline_matrix, new_matrix)

    def test_min_samples_in_node_one_matches_baseline(self):
        """A threshold of 1 must be a no-op: every leaf already has >=1 sample."""
        dist_baseline = DistanceRandomForestProximity()
        dist_baseline.calculate_terminals(estimator=self.model, X=self.X)
        baseline_matrix, _ = dist_baseline.calculate_distance_matrix(
            sample_indices=None
        )

        dist_new = DistanceRandomForestProximity(min_samples_in_node=1)
        dist_new.calculate_terminals(estimator=self.model, X=self.X)
        new_matrix, _ = dist_new.calculate_distance_matrix(sample_indices=None)

        np.testing.assert_array_equal(baseline_matrix, new_matrix)

    def test_min_samples_in_node_large_collapses_to_root(self):
        """A threshold larger than any node forces every leaf to the root -> all zeros."""
        huge = 10 * len(self.X)
        dist = DistanceRandomForestProximity(min_samples_in_node=huge)
        dist.calculate_terminals(estimator=self.model, X=self.X)
        matrix, _ = dist.calculate_distance_matrix(sample_indices=None)
        self.assertTrue(np.all(matrix == 0.0))

    def test_min_samples_in_node_monotonicity(self):
        """Mean off-diagonal distance is non-increasing as the threshold grows."""
        means = []
        for threshold in [1, 3, 5, 10, 25]:
            dist = DistanceRandomForestProximity(min_samples_in_node=threshold)
            dist.calculate_terminals(estimator=self.model, X=self.X)
            matrix, _ = dist.calculate_distance_matrix(sample_indices=None)
            n = matrix.shape[0]
            off_diag = matrix[~np.eye(n, dtype=bool)]
            means.append(off_diag.mean())
        for first, second in zip(means, means[1:]):
            self.assertLessEqual(second, first + 1e-8)

    def test_min_samples_in_node_invalid_raises(self):
        """Zero / negative thresholds are rejected at construction."""
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(min_samples_in_node=0)
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(min_samples_in_node=-5)

    def test_min_samples_in_node_preserves_shape_and_symmetry(self):
        """Collapsed matrix keeps the symmetric / zero-diagonal contract."""
        dist = DistanceRandomForestProximity(min_samples_in_node=5)
        dist.calculate_terminals(estimator=self.model, X=self.X)
        matrix, _ = dist.calculate_distance_matrix(sample_indices=None)
        self.assertEqual(matrix.shape, (len(self.X), len(self.X)))
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.all(np.diag(matrix) == 0))


class TestDistanceWasserstein(unittest.TestCase):
    def setUp(self):
        self.distance = DistanceWasserstein(scale_features=False)

    def test_calculate_distance_cluster_vs_background_continuous(self):
        bg = np.array([0, 1, 2, 3])
        cl = np.array([1, 2, 3, 4])

        result = self.distance.calculate_distance_cluster_vs_background(
            values_background=bg, values_cluster=cl, is_categorical=False
        )
        self.assertGreaterEqual(result, 0.0)

    def test_calculate_distance_cluster_vs_background_continuous_identical_distributions(
        self,
    ):
        dist = np.random.normal(loc=0.0, scale=1.0, size=100)

        result = self.distance.calculate_distance_cluster_vs_background(
            values_background=dist, values_cluster=dist, is_categorical=False
        )
        self.assertAlmostEqual(result, 0.0)

    def test_calculate_distance_cluster_vs_background_categorical(self):
        bg = pd.Series(data=["A", "A", "B", "C", "C"])
        cl = pd.Series(data=["A", "B", "B", "C"])

        result = self.distance.calculate_distance_cluster_vs_background(
            values_background=bg, values_cluster=cl, is_categorical=True
        )
        self.assertGreaterEqual(result, 0.0)

    def test_calculate_distance_cluster_vs_background_categorical_missing_category(
        self,
    ):
        bg = pd.Series(data=["A", "B", "C", "D"])
        cl = pd.Series(data=["A", "A", "B"])  # missing C and D

        result = self.distance.calculate_distance_cluster_vs_background(
            values_background=bg, values_cluster=cl, is_categorical=True
        )
        self.assertGreaterEqual(result, 0.0)

    def test_calculate_distance_cluster_vs_background_categorical_identical_distributions(
        self,
    ):
        bg = pd.Series(data=["X", "Y", "Z"] * 10)
        cl = pd.Series(data=["X", "Y", "Z"] * 10)

        result = self.distance.calculate_distance_cluster_vs_background(
            values_background=bg, values_cluster=cl, is_categorical=True
        )
        self.assertAlmostEqual(result, 0.0)


class TestDistanceJensenShannon(unittest.TestCase):
    def setUp(self):
        self.distance = DistanceJensenShannon(scale_features=False)

        np.random.seed(0)

    def test_calculate_distance_cluster_vs_background_continuous(self):
        bg = pd.Series(data=np.random.normal(loc=0, scale=1, size=1000))
        cl = pd.Series(data=np.random.normal(loc=0, scale=1, size=1000))

        result = self.distance.calculate_distance_cluster_vs_background(
            values_background=bg, values_cluster=cl, is_categorical=False
        )
        self.assertGreaterEqual(result, 0.0)

    def test_calculate_distance_cluster_vs_background_continuous_identical_distributions(
        self,
    ):
        dist = pd.Series(data=np.random.uniform(low=0, high=1, size=500))

        result = self.distance.calculate_distance_cluster_vs_background(
            values_background=dist, values_cluster=dist, is_categorical=False
        )
        self.assertAlmostEqual(result, 0.0)

    def test_calculate_distance_cluster_vs_background_categorical(self):
        bg = pd.Series(data=["A", "A", "B", "C", "C", "C"])
        cl = pd.Series(data=["A", "B", "B", "C"])

        result = self.distance.calculate_distance_cluster_vs_background(
            values_background=bg, values_cluster=cl, is_categorical=True
        )
        self.assertGreaterEqual(result, 0.0)

    def test_calculate_distance_cluster_vs_background_categorical_missing_category(
        self,
    ):
        bg = pd.Series(data=["A", "B", "C", "D"])
        cl = pd.Series(data=["A", "A", "B"])  # missing C and D

        result = self.distance.calculate_distance_cluster_vs_background(
            values_background=bg, values_cluster=cl, is_categorical=True
        )
        self.assertGreaterEqual(result, 0.0)

    def test_calculate_distance_cluster_vs_background_categorical_identical_distributions(
        self,
    ):
        bg = pd.Series(data=["X", "Y", "Z"] * 10)
        cl = pd.Series(data=["X", "Y", "Z"] * 10)

        result = self.distance.calculate_distance_cluster_vs_background(
            values_background=bg, values_cluster=cl, is_categorical=True
        )
        self.assertAlmostEqual(result, 0.0)
