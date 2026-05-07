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

    def _train_regression_model(self, criterion="squared_error"):
        from sklearn.datasets import make_regression
        from sklearn.ensemble import RandomForestRegressor

        X, y = make_regression(
            n_samples=50,
            n_features=5,
            n_informative=3,
            random_state=self.random_state,
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        model = RandomForestRegressor(
            n_estimators=20,
            max_depth=8,
            criterion=criterion,
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

    def test_max_depth_for_proximity_none_matches_baseline(self):
        """Default behavior (max_depth_for_proximity=None) must match pre-feature output."""
        dist_baseline = DistanceRandomForestProximity()
        dist_baseline.calculate_terminals(estimator=self.model, X=self.X)
        baseline_matrix, _ = dist_baseline.calculate_distance_matrix(
            sample_indices=None
        )

        dist_new = DistanceRandomForestProximity(max_depth_for_proximity=None)
        dist_new.calculate_terminals(estimator=self.model, X=self.X)
        new_matrix, _ = dist_new.calculate_distance_matrix(sample_indices=None)

        np.testing.assert_array_equal(baseline_matrix, new_matrix)

    def test_max_depth_for_proximity_zero_collapses_to_root(self):
        """A threshold of 0 forces every leaf to the root -> distance matrix is all zeros."""
        dist = DistanceRandomForestProximity(max_depth_for_proximity=0)
        dist.calculate_terminals(estimator=self.model, X=self.X)
        matrix, _ = dist.calculate_distance_matrix(sample_indices=None)
        self.assertTrue(np.all(matrix == 0.0))

    def test_max_depth_for_proximity_large_matches_baseline(self):
        """A very large threshold leaves every leaf untouched -> identical to baseline."""
        dist_baseline = DistanceRandomForestProximity()
        dist_baseline.calculate_terminals(estimator=self.model, X=self.X)
        baseline_matrix, _ = dist_baseline.calculate_distance_matrix(
            sample_indices=None
        )

        dist_new = DistanceRandomForestProximity(max_depth_for_proximity=10_000)
        dist_new.calculate_terminals(estimator=self.model, X=self.X)
        new_matrix, _ = dist_new.calculate_distance_matrix(sample_indices=None)

        np.testing.assert_array_equal(baseline_matrix, new_matrix)

    def test_max_depth_for_proximity_monotonicity(self):
        """Mean off-diagonal distance is non-decreasing as the depth threshold grows."""
        means = []
        for threshold in [0, 1, 2, 3, 5, 10, 100]:
            dist = DistanceRandomForestProximity(max_depth_for_proximity=threshold)
            dist.calculate_terminals(estimator=self.model, X=self.X)
            matrix, _ = dist.calculate_distance_matrix(sample_indices=None)
            n = matrix.shape[0]
            off_diag = matrix[~np.eye(n, dtype=bool)]
            means.append(float(off_diag.mean()))
        for a, b in zip(means, means[1:]):
            self.assertGreaterEqual(b, a - 1e-8)

    def test_max_depth_for_proximity_invalid_raises(self):
        """Negative thresholds are rejected at construction; 0 is allowed."""
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(max_depth_for_proximity=-1)
        DistanceRandomForestProximity(max_depth_for_proximity=0)

    def test_max_depth_for_proximity_preserves_shape_and_symmetry(self):
        """Collapsed matrix keeps the symmetric / zero-diagonal contract."""
        dist = DistanceRandomForestProximity(max_depth_for_proximity=3)
        dist.calculate_terminals(estimator=self.model, X=self.X)
        matrix, _ = dist.calculate_distance_matrix(sample_indices=None)
        self.assertEqual(matrix.shape, (len(self.X), len(self.X)))
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.all(np.diag(matrix) == 0))

    def test_min_samples_and_max_depth_mutually_exclusive(self):
        """Setting both ancestor-collapse parameters at once must raise ValueError."""
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(
                min_samples_in_node=5,
                max_depth_for_proximity=3,
            )

    def test_min_node_variance_none_matches_baseline(self):
        """Default behavior (min_node_variance=None) on a regressor matches baseline."""
        X_reg, _, model_reg = self._train_regression_model()
        dist_baseline = DistanceRandomForestProximity()
        dist_baseline.calculate_terminals(estimator=model_reg, X=X_reg)
        baseline_matrix, _ = dist_baseline.calculate_distance_matrix(
            sample_indices=None
        )

        dist_new = DistanceRandomForestProximity(min_node_variance=None)
        dist_new.calculate_terminals(estimator=model_reg, X=X_reg)
        new_matrix, _ = dist_new.calculate_distance_matrix(sample_indices=None)

        np.testing.assert_array_equal(baseline_matrix, new_matrix)

    def test_min_node_variance_zero_matches_baseline(self):
        """`min_node_variance=0` prunes nothing (every node has impurity >= 0) -> baseline."""
        X_reg, _, model_reg = self._train_regression_model()

        dist_baseline = DistanceRandomForestProximity()
        dist_baseline.calculate_terminals(estimator=model_reg, X=X_reg)
        baseline_matrix, _ = dist_baseline.calculate_distance_matrix(
            sample_indices=None
        )

        dist_new = DistanceRandomForestProximity(min_node_variance=0.0)
        dist_new.calculate_terminals(estimator=model_reg, X=X_reg)
        new_matrix, _ = dist_new.calculate_distance_matrix(sample_indices=None)

        np.testing.assert_array_equal(baseline_matrix, new_matrix)

    def test_min_node_variance_large_collapses_to_root(self):
        """A very large variance threshold collapses any leaf to the root."""
        X_reg, _, model_reg = self._train_regression_model()
        dist = DistanceRandomForestProximity(min_node_variance=10_000.0)
        dist.calculate_terminals(estimator=model_reg, X=X_reg)
        matrix, _ = dist.calculate_distance_matrix(sample_indices=None)
        self.assertEqual(matrix.shape, (len(X_reg), len(X_reg)))
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.all(np.diag(matrix) == 0))
        self.assertGreaterEqual(matrix.min(), 0.0 - 1e-6)
        self.assertLessEqual(matrix.max(), 1.0 + 1e-6)

    def test_min_node_variance_rejects_classifier(self):
        """Using min_node_variance with a classifier raises ValueError at calculate_terminals."""
        dist = DistanceRandomForestProximity(min_node_variance=1.0)
        with self.assertRaises(ValueError) as ctx:
            dist.calculate_terminals(estimator=self.model, X=self.X)
        self.assertIn("RandomForestRegressor", str(ctx.exception))

    def test_min_node_variance_rejects_absolute_error_criterion(self):
        """Using min_node_variance with criterion 'absolute_error' raises ValueError."""
        X_reg, _, model_reg = self._train_regression_model(criterion="absolute_error")
        dist = DistanceRandomForestProximity(min_node_variance=1.0)
        with self.assertRaises(ValueError) as ctx:
            dist.calculate_terminals(estimator=model_reg, X=X_reg)
        self.assertIn("squared_error", str(ctx.exception))
        self.assertIn("friedman_mse", str(ctx.exception))

    def test_min_node_variance_invalid_raises(self):
        """Negative thresholds are rejected at construction; 0 is allowed."""
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(min_node_variance=-0.5)
        DistanceRandomForestProximity(min_node_variance=0.0)

    def test_min_node_variance_mutually_exclusive_with_min_samples(self):
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(min_samples_in_node=5, min_node_variance=1.0)

    def test_min_node_variance_mutually_exclusive_with_max_depth(self):
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(
                max_depth_for_proximity=3, min_node_variance=1.0
            )

    def test_min_node_variance_all_three_mutually_exclusive(self):
        """Setting any two of the three collapse params at once must raise."""
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(
                min_samples_in_node=5,
                max_depth_for_proximity=3,
                min_node_variance=1.0,
            )

    def test_min_node_variance_preserves_shape_and_symmetry(self):
        """Collapsed matrix keeps the symmetric / zero-diagonal contract on a regressor."""
        X_reg, _, model_reg = self._train_regression_model()
        dist = DistanceRandomForestProximity(min_node_variance=0.5)
        dist.calculate_terminals(estimator=model_reg, X=X_reg)
        matrix, _ = dist.calculate_distance_matrix(sample_indices=None)
        self.assertEqual(matrix.shape, (len(X_reg), len(X_reg)))
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.all(np.diag(matrix) == 0))

    def test_min_node_variance_friedman_mse_emits_warning_and_proceeds(self):
        """Using min_node_variance with criterion='friedman_mse' must warn and complete."""
        import warnings

        X_reg, _, model_reg = self._train_regression_model(criterion="friedman_mse")
        dist = DistanceRandomForestProximity(min_node_variance=1.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dist.calculate_terminals(estimator=model_reg, X=X_reg)

        matched = [
            w
            for w in caught
            if issubclass(w.category, UserWarning) and "friedman_mse" in str(w.message)
        ]
        self.assertEqual(
            len(matched),
            1,
            msg=f"Expected exactly one friedman_mse UserWarning, got {len(matched)}",
        )
        self.assertIsNotNone(dist.terminals)
        self.assertEqual(dist.terminals.shape, (len(X_reg), model_reg.n_estimators))


class TestDistanceRandomForestLCA(unittest.TestCase):
    def setUp(self):
        self.tmp_path = os.path.join(os.getcwd(), "tmp_fgc_lca")
        Path(self.tmp_path).mkdir(parents=True, exist_ok=True)

        self.random_state = 42
        self.X, self.y, self.model = self._train_regression_model()

    def _train_regression_model(self):
        from sklearn.datasets import make_regression
        from sklearn.ensemble import RandomForestRegressor

        X, y = make_regression(
            n_samples=50,
            n_features=5,
            n_informative=3,
            random_state=self.random_state,
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        model = RandomForestRegressor(
            n_estimators=20,
            max_depth=8,
            random_state=self.random_state,
        )
        model.fit(X=X, y=y)
        return X, y, model

    def tearDown(self):
        try:
            shutil.rmtree(self.tmp_path)
        except OSError:
            pass

    def test_calculate_terminals_populates_state(self):
        from fgclustering.distance import DistanceRandomForestLCA, _compute_node_depths

        dist = DistanceRandomForestLCA()
        dist.calculate_terminals(estimator=self.model, X=self.X)
        self.assertIsNotNone(dist.terminals)
        self.assertIsNotNone(dist.paths)
        self.assertIsNotNone(dist.path_lens)
        self.assertEqual(dist.terminals.shape, (len(self.X), self.model.n_estimators))
        self.assertEqual(dist.paths.shape[0], len(self.X))
        self.assertEqual(dist.paths.shape[1], self.model.n_estimators)
        self.assertEqual(dist.path_lens.shape, (len(self.X), self.model.n_estimators))
        self.assertTrue(np.all(dist.paths[:, :, 0] == 0))
        for t, dt in enumerate(self.model.estimators_):
            tree = dt.tree_
            depths = _compute_node_depths(tree)
            expected_lens = depths[dist.terminals[:, t]] + 1
            np.testing.assert_array_equal(dist.path_lens[:, t], expected_lens)

    def test_calculate_distance_matrix_shape_symmetry_and_diagonal(self):
        from fgclustering.distance import DistanceRandomForestLCA

        dist = DistanceRandomForestLCA()
        dist.calculate_terminals(estimator=self.model, X=self.X)
        matrix, file = dist.calculate_distance_matrix(sample_indices=None)
        self.assertEqual(matrix.shape, (len(self.X), len(self.X)))
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.all(np.diag(matrix) == 0))
        self.assertIsNone(file)

    def test_same_leaf_samples_have_zero_distance(self):
        """Samples that share the same leaf in every tree must have distance 0."""
        from fgclustering.distance import DistanceRandomForestLCA

        dist = DistanceRandomForestLCA()
        dist.calculate_terminals(estimator=self.model, X=self.X)
        same_terminals = (dist.terminals[:, None, :] == dist.terminals[None, :, :]).all(
            axis=2
        )
        matrix, _ = dist.calculate_distance_matrix(sample_indices=None)
        self.assertTrue(np.all(matrix[same_terminals] == 0.0))

    def test_memory_efficient_memmap_path_matches_in_memory(self):
        from fgclustering.distance import DistanceRandomForestLCA

        d1 = DistanceRandomForestLCA(memory_efficient=False)
        d1.calculate_terminals(estimator=self.model, X=self.X)
        m1, _ = d1.calculate_distance_matrix(sample_indices=None)

        d2 = DistanceRandomForestLCA(
            memory_efficient=True, dir_distance_matrix=self.tmp_path
        )
        d2.calculate_terminals(estimator=self.model, X=self.X)
        m2, f2 = d2.calculate_distance_matrix(sample_indices=None)

        self.assertTrue(isinstance(m2, np.memmap))
        self.assertTrue(os.path.exists(f2))
        np.testing.assert_allclose(np.asarray(m1), np.asarray(m2), atol=1e-6)

        d2.remove_distance_matrix(m2, f2)
        self.assertFalse(os.path.exists(f2))

    def test_calculate_distance_matrix_error_without_paths(self):
        from fgclustering.distance import DistanceRandomForestLCA

        dist = DistanceRandomForestLCA()
        with self.assertRaises(ValueError):
            dist.calculate_distance_matrix(sample_indices=None)

    def test_init_missing_dir_in_memory_efficient_mode(self):
        from fgclustering.distance import DistanceRandomForestLCA

        with self.assertRaises(ValueError):
            DistanceRandomForestLCA(memory_efficient=True)

    def test_sample_indices_slicing(self):
        from fgclustering.distance import DistanceRandomForestLCA

        dist = DistanceRandomForestLCA()
        dist.calculate_terminals(estimator=self.model, X=self.X)
        idx = np.random.RandomState(0).choice(len(self.X), size=20, replace=False)
        matrix, _ = dist.calculate_distance_matrix(sample_indices=idx)
        self.assertEqual(matrix.shape, (20, 20))
        self.assertTrue(np.allclose(matrix, matrix.T))

    def test_lca_distance_increases_when_deeper_path_is_grown(self):
        """Growing the deeper path lowers similarity and increases distance under max-based normalization."""
        from fgclustering.distance import _calculate_lca_distances

        n_trees = 1

        def run(paths_list, lens_list):
            max_len = max(len(path) for path in paths_list)
            paths_arr = np.full((2, n_trees, max_len), -1, dtype=np.int32)
            for idx, path in enumerate(paths_list):
                paths_arr[idx, 0, : len(path)] = path
            lens_arr = np.array([[lens_list[0]], [lens_list[1]]], dtype=np.int32)
            out = np.zeros((2, 2), dtype=np.float32)
            return _calculate_lca_distances(paths_arr, lens_arr, 2, n_trees, out)

        case_a = run([[0, 1], [0, 1, 2]], [2, 3])
        case_b = run([[0, 1], [0, 1, 2, 3]], [2, 4])

        self.assertAlmostEqual(float(case_a[0, 1]), 0.5, places=5)
        self.assertAlmostEqual(float(case_b[0, 1]), 2 / 3, places=5)

    def test_lca_distance_root_only_trees_are_treated_as_identical(self):
        """Both samples at depth 0 fall back to similarity 1.0."""
        from fgclustering.distance import _calculate_lca_distances

        paths = np.full((2, 1, 1), -1, dtype=np.int32)
        paths[0, 0, 0] = 0
        paths[1, 0, 0] = 0
        lens = np.ones((2, 1), dtype=np.int32)
        out = np.zeros((2, 2), dtype=np.float32)
        result = _calculate_lca_distances(paths, lens, 2, 1, out)
        self.assertEqual(float(result[0, 1]), 0.0)


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
