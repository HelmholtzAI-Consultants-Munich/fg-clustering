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

from fgclustering.distance import DistanceRandomForestProximity, DistanceWasserstein, DistanceJensenShannon

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
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y = y

        model = RandomForestClassifier(
            max_depth=10,
            max_features="sqrt",
            max_samples=0.8,
            bootstrap=True,
            oob_score=True,
            random_state=self.random_state,
        )
        model.fit(X, y)

        return X, y, model

    def tearDown(self):
        try:
            shutil.rmtree(self.tmp_path)
        except:
            pass

    def test_calculate_terminals(self):
        dist = DistanceRandomForestProximity()
        dist.calculate_terminals(self.model, self.X)
        self.assertIsNotNone(dist.terminals)
        self.assertEqual(dist.terminals.shape[0], self.X.shape[0])

    def test_calculate_distance_matrix_non_memory_efficient(self):
        dist = DistanceRandomForestProximity(memory_efficient=False)
        dist.calculate_terminals(self.model, self.X)
        matrix = dist.calculate_distance_matrix(sample_indices=None)
        self.assertEqual(matrix.shape[0], matrix.shape[1])
        self.assertEqual(matrix.shape[0], len(self.X))
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.all(np.diag(matrix) == 0))

    def test_calculate_distance_matrix_memory_efficient(self):
        dist = DistanceRandomForestProximity(memory_efficient=True, dir_distance_matrix=self.tmp_path)
        dist.calculate_terminals(self.model, self.X)
        matrix = dist.calculate_distance_matrix(sample_indices=None)
        self.assertTrue(isinstance(matrix, np.memmap))
        self.assertEqual(matrix.shape[0], matrix.shape[1])
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.all(np.diag(matrix) == 0))
        self.assertTrue(os.path.exists(dist.file_distance_matrix))

    def test_calculate_distance_matrix_error_without_terminals(self):
        dist = DistanceRandomForestProximity()
        with self.assertRaises(ValueError):
            _ = dist.calculate_distance_matrix(sample_indices=None)

    def test_calculate_distance_matrix_error_missing_dir_in_memory_efficient_mode(self):
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(memory_efficient=True)

    def test_calculate_distance_matrix_with_sample_indices(self):
        dist = DistanceRandomForestProximity(memory_efficient=False)
        dist.calculate_terminals(self.model, self.X)
        sample_indices = np.random.choice(len(self.X), size=20, replace=False)
        matrix = dist.calculate_distance_matrix(sample_indices=sample_indices)
        self.assertEqual(matrix.shape, (20, 20))


class TestDistanceWasserstein(unittest.TestCase):
    def setUp(self):
        self.distance = DistanceWasserstein(scale_features=False)

    def test_calculate_distance_cluster_vs_background_continuous(self):
        bg = np.array([0, 1, 2, 3])
        cl = np.array([1, 2, 3, 4])

        result = self.distance.calculate_distance_cluster_vs_background(bg, cl, is_categorical=False)
        self.assertGreaterEqual(result, 0.0)

    def test_calculate_distance_cluster_vs_background_continuous_identical_distributions(self):
        dist = np.random.normal(size=100)

        result = self.distance.calculate_distance_cluster_vs_background(dist, dist, is_categorical=False)
        self.assertAlmostEqual(result, 0.0)

    def test_calculate_distance_cluster_vs_background_categorical(self):
        bg = pd.Series(["A", "A", "B", "C", "C"])
        cl = pd.Series(["A", "B", "B", "C"])

        result = self.distance.calculate_distance_cluster_vs_background(bg, cl, is_categorical=True)
        self.assertGreaterEqual(result, 0.0)

    def test_calculate_distance_cluster_vs_background_categorical_missing_category(self):
        bg = pd.Series(["A", "B", "C", "D"])
        cl = pd.Series(["A", "A", "B"])  # missing C and D

        result = self.distance.calculate_distance_cluster_vs_background(bg, cl, is_categorical=True)
        self.assertGreaterEqual(result, 0.0)

    def test_calculate_distance_cluster_vs_background_categorical_identical_distributions(self):
        bg = pd.Series(["X", "Y", "Z"] * 10)
        cl = pd.Series(["X", "Y", "Z"] * 10)

        result = self.distance.calculate_distance_cluster_vs_background(bg, cl, is_categorical=True)
        self.assertAlmostEqual(result, 0.0)


class TestDistanceJensenShannon(unittest.TestCase):
    def setUp(self):
        self.distance = DistanceJensenShannon(scale_features=False)

        np.random.seed(0)

    def test_calculate_distance_cluster_vs_background_continuous(self):
        bg = pd.Series(np.random.normal(0, 1, 1000))
        cl = pd.Series(np.random.normal(0, 1, 1000))

        result = self.distance.calculate_distance_cluster_vs_background(bg, cl, is_categorical=False)
        self.assertGreaterEqual(result, 0.0)

    def test_calculate_distance_cluster_vs_background_continuous_identical_distributions(self):
        dist = pd.Series(np.random.uniform(0, 1, 500))

        result = self.distance.calculate_distance_cluster_vs_background(dist, dist, is_categorical=False)
        self.assertAlmostEqual(result, 0.0)

    def test_calculate_distance_cluster_vs_background_categorical(self):
        bg = pd.Series(["A", "A", "B", "C", "C", "C"])
        cl = pd.Series(["A", "B", "B", "C"])

        result = self.distance.calculate_distance_cluster_vs_background(bg, cl, is_categorical=True)
        self.assertGreaterEqual(result, 0.0)

    def test_calculate_distance_cluster_vs_background_categorical_missing_category(self):
        bg = pd.Series(["A", "B", "C", "D"])
        cl = pd.Series(["A", "A", "B"])  # missing C and D

        result = self.distance.calculate_distance_cluster_vs_background(bg, cl, is_categorical=True)
        self.assertGreaterEqual(result, 0.0)

    def test_calculate_distance_cluster_vs_background_categorical_identical_distributions(self):
        bg = pd.Series(["X", "Y", "Z"] * 10)
        cl = pd.Series(["X", "Y", "Z"] * 10)

        result = self.distance.calculate_distance_cluster_vs_background(bg, cl, is_categorical=True)
        self.assertAlmostEqual(result, 0.0)
