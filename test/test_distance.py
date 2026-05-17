############################################
# imports
############################################

import os
import shutil
import unittest
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from fgclustering.distance import (
    DistanceRandomForestBase,
    DistanceRandomForestLCA,
    DistanceRandomForestProximity,
    DistanceWasserstein,
    DistanceJensenShannon,
    _build_leaf_to_ancestor_map,
    _calculate_distance_lca,
    _compute_node_depths,
    _compute_parent_array,
    _validate_mutually_exclusive,
)

############################################
# Tests
############################################


def _build_regression_forest(
    random_state=42,
    criterion="squared_error",
    n_estimators=20,
    max_depth=8,
):
    X, y = make_regression(
        n_samples=50,
        n_features=5,
        n_informative=3,
        random_state=random_state,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion=criterion,
        random_state=random_state,
    ).fit(X, y)
    return X, y, model


def _build_classification_forest(
    random_state=42,
    max_depth=10,
    max_features="sqrt",
    max_samples=0.8,
    bootstrap=True,
    oob_score=True,
):
    # Generate test data
    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=random_state,
    )
    X = pd.DataFrame(data=X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    model = RandomForestClassifier(
        max_depth=max_depth,
        max_features=max_features,
        max_samples=max_samples,
        bootstrap=bootstrap,
        oob_score=oob_score,
        random_state=random_state,
    ).fit(X, y)
    return X, y, model


class TestDistanceRandomForestProximity(unittest.TestCase):
    def setUp(self):
        self.tmp_path = os.path.join(os.getcwd(), "tmp_fgc")
        Path(self.tmp_path).mkdir(parents=True, exist_ok=True)

        self.random_state = 42
        self.n_jobs = 2
        self.verbose = 1

        self.X_clas, self.y_clas, self.model_clas = self._train_classification_model()
        self.X_reg_squared, self.y_reg_squared, self.model_reg_squared = self._train_regression_model(
            criterion="squared_error"
        )
        self.X_reg_friedman, self.y_reg_friedman, self.model_reg_friedman = self._train_regression_model(
            criterion="friedman_mse"
        )
        self.X_reg_absolute, self.y_reg_absolute, self.model_reg_absolute = self._train_regression_model(
            criterion="absolute_error"
        )

    def _train_classification_model(self):
        return _build_classification_forest(
            random_state=self.random_state,
        )

    def _train_regression_model(self, criterion="squared_error"):
        return _build_regression_forest(
            random_state=self.random_state,
            criterion=criterion,
        )

    def tearDown(self):
        try:
            shutil.rmtree(self.tmp_path)
        except OSError:
            pass

    def test_calculate_forest_encoding(self):
        dist = DistanceRandomForestProximity()
        dist.calculate_forest_encoding(estimator=self.model_clas, X=self.X_clas)
        self.assertIsNotNone(dist.terminals)
        self.assertEqual(dist.terminals.shape[0], self.X_clas.shape[0])

    def test_calculate_distance_matrix_non_memory_efficient(self):
        dist = DistanceRandomForestProximity(memory_efficient=False)
        dist.calculate_forest_encoding(estimator=self.model_clas, X=self.X_clas)
        matrix, file = dist.calculate_distance_matrix(sample_indices=None)
        self.assertEqual(matrix.shape[0], matrix.shape[1])
        self.assertEqual(matrix.shape[0], len(self.X_clas))
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.all(np.diag(matrix) == 0))
        self.assertTrue(file is None)

    def test_calculate_distance_matrix_memory_efficient(self):
        dist = DistanceRandomForestProximity(memory_efficient=True, dir_distance_matrix=self.tmp_path)
        dist.calculate_forest_encoding(estimator=self.model_clas, X=self.X_clas)
        matrix, file = dist.calculate_distance_matrix(sample_indices=None)
        self.assertTrue(isinstance(matrix, np.memmap))
        self.assertEqual(matrix.shape[0], matrix.shape[1])
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.all(np.diag(matrix) == 0))
        self.assertTrue(os.path.exists(file))

    def test_calculate_distance_matrix_error_without_terminals(self):
        dist = DistanceRandomForestProximity()
        with self.assertRaises(ValueError):
            _ = dist.calculate_distance_matrix(sample_indices=None)

    def test_calculate_distance_matrix_error_missing_dir_in_memory_efficient_mode(self):
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(memory_efficient=True)

    def test_calculate_distance_matrix_with_sample_indices(self):
        dist = DistanceRandomForestProximity(memory_efficient=False)
        dist.calculate_forest_encoding(estimator=self.model_clas, X=self.X_clas)
        sample_indices = np.random.choice(len(self.X_clas), size=20, replace=False)
        matrix, file = dist.calculate_distance_matrix(sample_indices=sample_indices)
        self.assertEqual(matrix.shape, (20, 20))

    def test_min_samples_in_node_none_matches_baseline(self):
        """Default behavior (min_samples_in_node=None) must match pre-feature output."""
        dist_baseline = DistanceRandomForestProximity()
        dist_baseline.calculate_forest_encoding(estimator=self.model_clas, X=self.X_clas)
        baseline_matrix, _ = dist_baseline.calculate_distance_matrix(sample_indices=None)

        dist_new = DistanceRandomForestProximity(min_samples_in_node=None)
        dist_new.calculate_forest_encoding(estimator=self.model_clas, X=self.X_clas)
        new_matrix, _ = dist_new.calculate_distance_matrix(sample_indices=None)

        np.testing.assert_array_equal(baseline_matrix, new_matrix)

    def test_min_samples_in_node_one_matches_baseline(self):
        """A threshold of 1 must be a no-op: every leaf already has >=1 sample."""
        dist_baseline = DistanceRandomForestProximity()
        dist_baseline.calculate_forest_encoding(estimator=self.model_clas, X=self.X_clas)
        baseline_matrix, _ = dist_baseline.calculate_distance_matrix(sample_indices=None)

        dist_new = DistanceRandomForestProximity(min_samples_in_node=1)
        dist_new.calculate_forest_encoding(estimator=self.model_clas, X=self.X_clas)
        new_matrix, _ = dist_new.calculate_distance_matrix(sample_indices=None)

        np.testing.assert_array_equal(baseline_matrix, new_matrix)

        baseline = DistanceRandomForestProximity()
        baseline.calculate_forest_encoding(estimator=self.model_reg_squared, X=self.X_reg_squared)
        bm, _ = baseline.calculate_distance_matrix(sample_indices=None)

        new = DistanceRandomForestProximity(min_samples_in_node=1)
        new.calculate_forest_encoding(estimator=self.model_reg_squared, X=self.X_reg_squared)
        nm, _ = new.calculate_distance_matrix(sample_indices=None)
        np.testing.assert_array_equal(bm, nm)

    def test_min_samples_in_node_large_collapses_to_root(self):
        """A threshold larger than any node forces every leaf to the root -> all zeros."""
        huge = 10 * len(self.X_clas)
        dist = DistanceRandomForestProximity(min_samples_in_node=huge)
        dist.calculate_forest_encoding(estimator=self.model_clas, X=self.X_clas)
        matrix, _ = dist.calculate_distance_matrix(sample_indices=None)
        self.assertTrue(np.all(matrix == 0.0))

    def test_min_samples_in_node_invalid_raises(self):
        """Zero / negative thresholds are rejected at construction."""
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(min_samples_in_node=0)
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(min_samples_in_node=-5)

    def test_min_samples_in_node_preserves_shape_and_symmetry(self):
        """Collapsed matrix keeps the symmetric / zero-diagonal contract."""
        dist = DistanceRandomForestProximity(min_samples_in_node=5)
        dist.calculate_forest_encoding(estimator=self.model_clas, X=self.X_clas)
        matrix, _ = dist.calculate_distance_matrix(sample_indices=None)
        self.assertEqual(matrix.shape, (len(self.X_clas), len(self.X_clas)))
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.all(np.diag(matrix) == 0))

    def test_max_depth_for_proximity_none_matches_baseline(self):
        """Default behavior (max_depth_for_proximity=None) must match pre-feature output."""
        dist_baseline = DistanceRandomForestProximity()
        dist_baseline.calculate_forest_encoding(estimator=self.model_clas, X=self.X_clas)
        baseline_matrix, _ = dist_baseline.calculate_distance_matrix(sample_indices=None)

        dist_new = DistanceRandomForestProximity(max_depth_for_proximity=None)
        dist_new.calculate_forest_encoding(estimator=self.model_clas, X=self.X_clas)
        new_matrix, _ = dist_new.calculate_distance_matrix(sample_indices=None)

        np.testing.assert_array_equal(baseline_matrix, new_matrix)

    def test_max_depth_for_proximity_zero_collapses_to_root(self):
        """A threshold of 0 forces every leaf to the root -> distance matrix is all zeros."""
        dist = DistanceRandomForestProximity(max_depth_for_proximity=0)
        dist.calculate_forest_encoding(estimator=self.model_clas, X=self.X_clas)
        matrix, _ = dist.calculate_distance_matrix(sample_indices=None)
        self.assertTrue(np.all(matrix == 0.0))

    def test_max_depth_for_proximity_large_matches_baseline(self):
        """A very large threshold leaves every leaf untouched -> identical to baseline."""
        dist_baseline = DistanceRandomForestProximity()
        dist_baseline.calculate_forest_encoding(estimator=self.model_clas, X=self.X_clas)
        baseline_matrix, _ = dist_baseline.calculate_distance_matrix(sample_indices=None)

        dist_new = DistanceRandomForestProximity(max_depth_for_proximity=10_000)
        dist_new.calculate_forest_encoding(estimator=self.model_clas, X=self.X_clas)
        new_matrix, _ = dist_new.calculate_distance_matrix(sample_indices=None)

        np.testing.assert_array_equal(baseline_matrix, new_matrix)

        baseline = DistanceRandomForestProximity()
        baseline.calculate_forest_encoding(estimator=self.model_reg_squared, X=self.X_reg_squared)
        bm, _ = baseline.calculate_distance_matrix(sample_indices=None)

        new = DistanceRandomForestProximity(max_depth_for_proximity=10_000)
        new.calculate_forest_encoding(estimator=self.model_reg_squared, X=self.X_reg_squared)
        nm, _ = new.calculate_distance_matrix(sample_indices=None)

        np.testing.assert_array_equal(bm, nm)

    def test_max_depth_for_proximity_invalid_raises(self):
        """Negative thresholds are rejected at construction; 0 is allowed."""
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(max_depth_for_proximity=-1)
        DistanceRandomForestProximity(max_depth_for_proximity=0)

    def test_max_depth_for_proximity_preserves_shape_and_symmetry(self):
        """Collapsed matrix keeps the symmetric / zero-diagonal contract."""
        dist = DistanceRandomForestProximity(max_depth_for_proximity=3)
        dist.calculate_forest_encoding(estimator=self.model_clas, X=self.X_clas)
        matrix, _ = dist.calculate_distance_matrix(sample_indices=None)
        self.assertEqual(matrix.shape, (len(self.X_clas), len(self.X_clas)))
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.all(np.diag(matrix) == 0))

    def test_min_variance_in_node_none_matches_baseline(self):
        """Default behavior (min_variance_in_node=None) on a regressor matches baseline."""
        dist_baseline = DistanceRandomForestProximity()
        dist_baseline.calculate_forest_encoding(estimator=self.model_reg_squared, X=self.X_reg_squared)
        baseline_matrix, _ = dist_baseline.calculate_distance_matrix(sample_indices=None)

        dist_new = DistanceRandomForestProximity(min_variance_in_node=None)
        dist_new.calculate_forest_encoding(estimator=self.model_reg_squared, X=self.X_reg_squared)
        new_matrix, _ = dist_new.calculate_distance_matrix(sample_indices=None)

        np.testing.assert_array_equal(baseline_matrix, new_matrix)

    def test_min_variance_in_node_zero_matches_baseline(self):
        """`min_variance_in_node=0` prunes nothing (every node has impurity >= 0) -> baseline."""
        dist_baseline = DistanceRandomForestProximity()
        dist_baseline.calculate_forest_encoding(estimator=self.model_reg_squared, X=self.X_reg_squared)
        baseline_matrix, _ = dist_baseline.calculate_distance_matrix(sample_indices=None)

        dist_new = DistanceRandomForestProximity(min_variance_in_node=0.0)
        dist_new.calculate_forest_encoding(estimator=self.model_reg_squared, X=self.X_reg_squared)
        new_matrix, _ = dist_new.calculate_distance_matrix(sample_indices=None)

        np.testing.assert_array_equal(baseline_matrix, new_matrix)

    def test_min_variance_in_node_large_collapses_to_root(self):
        """A very large variance threshold collapses any leaf to the root."""
        dist = DistanceRandomForestProximity(min_variance_in_node=10_000.0)
        dist.calculate_forest_encoding(estimator=self.model_reg_squared, X=self.X_reg_squared)
        matrix, _ = dist.calculate_distance_matrix(sample_indices=None)
        self.assertEqual(matrix.shape, (len(self.X_reg_squared), len(self.X_reg_squared)))
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.all(np.diag(matrix) == 0))
        self.assertGreaterEqual(matrix.min(), 0.0 - 1e-6)
        self.assertLessEqual(matrix.max(), 1.0 + 1e-6)

    def test_min_variance_in_node_rejects_classifier(self):
        """Using min_variance_in_node with a classifier raises ValueError at calculate_forest_encoding."""
        dist = DistanceRandomForestProximity(min_variance_in_node=1.0)
        with self.assertRaises(ValueError) as ctx:
            dist.calculate_forest_encoding(estimator=self.model_clas, X=self.X_clas)
        self.assertIn("RandomForestRegressor", str(ctx.exception))

    def test_min_variance_in_node_rejects_absolute_error_criterion(self):
        """Using min_variance_in_node with criterion 'absolute_error' raises ValueError."""
        dist = DistanceRandomForestProximity(min_variance_in_node=1.0)
        with self.assertRaises(ValueError) as ctx:
            dist.calculate_forest_encoding(estimator=self.model_reg_absolute, X=self.X_reg_absolute)
        self.assertIn("squared_error", str(ctx.exception))
        self.assertIn("friedman_mse", str(ctx.exception))

    def test_min_variance_in_node_invalid_raises(self):
        """Negative thresholds are rejected at construction; 0 is allowed."""
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(min_variance_in_node=-0.5)
        DistanceRandomForestProximity(min_variance_in_node=0.0)

    def test_min_variance_in_node_preserves_shape_and_symmetry(self):
        """Collapsed matrix keeps the symmetric / zero-diagonal contract on a regressor."""
        dist = DistanceRandomForestProximity(min_variance_in_node=0.5)
        dist.calculate_forest_encoding(estimator=self.model_reg_squared, X=self.X_reg_squared)
        matrix, _ = dist.calculate_distance_matrix(sample_indices=None)
        self.assertEqual(matrix.shape, (len(self.X_reg_squared), len(self.X_reg_squared)))
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.all(np.diag(matrix) == 0))

    def test_min_variance_in_node_friedman_mse_emits_warning_and_proceeds(self):
        """Using min_variance_in_node with criterion='friedman_mse' must warn and complete."""
        import warnings

        dist = DistanceRandomForestProximity(min_variance_in_node=1.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dist.calculate_forest_encoding(estimator=self.model_reg_friedman, X=self.X_reg_friedman)

        matched = [
            w for w in caught if issubclass(w.category, UserWarning) and "friedman_mse" in str(w.message)
        ]
        self.assertEqual(
            len(matched),
            1,
            msg=f"Expected exactly one friedman_mse UserWarning, got {len(matched)}",
        )
        self.assertIsNotNone(dist.terminals)
        self.assertEqual(
            dist.terminals.shape, (len(self.X_reg_friedman), self.model_reg_friedman.n_estimators)
        )

    def test_min_samples_mutually_exclusive_with_max_depth(self):
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(min_samples_in_node=5, max_depth_for_proximity=3)

    def test_min_samples_mutually_exclusive_with_min_variance_in_node(self):
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(min_samples_in_node=5, min_variance_in_node=1.0)

    def test_max_depth_mutually_exclusive_with_min_variance_in_node(self):
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(max_depth_for_proximity=3, min_variance_in_node=1.0)

    def test_all_three_mutually_exclusive(self):
        """Setting any two of the three collapse params at once must raise."""
        with self.assertRaises(ValueError):
            DistanceRandomForestProximity(
                min_samples_in_node=5,
                max_depth_for_proximity=3,
                min_variance_in_node=1.0,
            )


class TestDistanceRandomForestLCA(unittest.TestCase):
    def setUp(self):
        self.tmp_path = os.path.join(os.getcwd(), "tmp_fgc_lca")
        Path(self.tmp_path).mkdir(parents=True, exist_ok=True)

        self.random_state = 42
        self.X_reg, self.y_reg, self.model_reg = self._train_regression_model()

    def _train_regression_model(self, criterion="squared_error"):
        return _build_regression_forest(
            random_state=self.random_state,
            criterion=criterion,
        )

    def tearDown(self):
        try:
            shutil.rmtree(self.tmp_path)
        except OSError:
            pass

    def test_calculate_forest_encoding_populates_state(self):
        terminals = self.model_reg.apply(self.X_reg).astype(np.int32)

        dist = DistanceRandomForestLCA()
        dist.calculate_forest_encoding(estimator=self.model_reg, X=self.X_reg)
        self.assertIsNotNone(dist.paths)
        self.assertIsNotNone(dist.path_lens)
        self.assertEqual(terminals.shape, (len(self.X_reg), self.model_reg.n_estimators))
        self.assertEqual(dist.paths.shape[0], len(self.X_reg))
        self.assertEqual(dist.paths.shape[1], self.model_reg.n_estimators)
        self.assertEqual(dist.path_lens.shape, (len(self.X_reg), self.model_reg.n_estimators))
        self.assertTrue(np.all(dist.paths[:, :, 0] == 0))
        for t, dt in enumerate(self.model_reg.estimators_):
            tree = dt.tree_
            depths = _compute_node_depths(tree)
            expected_lens = depths[terminals[:, t]] + 1
            np.testing.assert_array_equal(dist.path_lens[:, t], expected_lens)

    def test_calculate_distance_matrix_shape_symmetry_and_diagonal(self):
        dist = DistanceRandomForestLCA()
        dist.calculate_forest_encoding(estimator=self.model_reg, X=self.X_reg)
        matrix, file = dist.calculate_distance_matrix(sample_indices=None)
        self.assertEqual(matrix.shape, (len(self.X_reg), len(self.X_reg)))
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.all(np.diag(matrix) == 0))
        self.assertIsNone(file)

    def test_memory_efficient_memmap_path_matches_in_memory(self):
        d1 = DistanceRandomForestLCA(memory_efficient=False)
        d1.calculate_forest_encoding(estimator=self.model_reg, X=self.X_reg)
        m1, _ = d1.calculate_distance_matrix(sample_indices=None)

        d2 = DistanceRandomForestLCA(memory_efficient=True, dir_distance_matrix=self.tmp_path)
        d2.calculate_forest_encoding(estimator=self.model_reg, X=self.X_reg)
        m2, f2 = d2.calculate_distance_matrix(sample_indices=None)

        self.assertTrue(isinstance(m2, np.memmap))
        self.assertTrue(os.path.exists(f2))
        np.testing.assert_allclose(np.asarray(m1), np.asarray(m2), atol=1e-6)

        d2.remove_distance_matrix(m2, f2)
        self.assertFalse(os.path.exists(f2))

    def test_calculate_distance_matrix_error_without_paths(self):
        dist = DistanceRandomForestLCA()
        with self.assertRaises(ValueError):
            dist.calculate_distance_matrix(sample_indices=None)

    def test_init_missing_dir_in_memory_efficient_mode(self):
        with self.assertRaises(ValueError):
            DistanceRandomForestLCA(memory_efficient=True)

    def test_sample_indices_slicing(self):
        dist = DistanceRandomForestLCA()
        dist.calculate_forest_encoding(estimator=self.model_reg, X=self.X_reg)
        idx = np.random.RandomState(0).choice(len(self.X_reg), size=20, replace=False)
        matrix, _ = dist.calculate_distance_matrix(sample_indices=idx)
        self.assertEqual(matrix.shape, (20, 20))
        self.assertTrue(np.allclose(matrix, matrix.T))

    def test_lca_distance_increases_when_deeper_path_is_grown(self):
        """Growing the deeper path lowers similarity and increases distance under max-based normalization."""
        n_trees = 1

        def run(paths_list, lens_list):
            max_len = max(len(path) for path in paths_list)
            paths_arr = np.full((2, n_trees, max_len), -1, dtype=np.int32)
            for idx, path in enumerate(paths_list):
                paths_arr[idx, 0, : len(path)] = path
            lens_arr = np.array([[lens_list[0]], [lens_list[1]]], dtype=np.int32)
            out = np.zeros((2, 2), dtype=np.float32)
            return _calculate_distance_lca(paths_arr, lens_arr, 2, n_trees, out)

        case_a = run([[0, 1], [0, 1, 2]], [2, 3])
        case_b = run([[0, 1], [0, 1, 2, 3]], [2, 4])

        self.assertAlmostEqual(float(case_a[0, 1]), 0.5, places=5)
        self.assertAlmostEqual(float(case_b[0, 1]), 2 / 3, places=5)

    def test_lca_distance_root_only_trees_are_treated_as_identical(self):
        """Both samples at depth 0 fall back to similarity 1.0."""
        paths = np.full((2, 1, 1), -1, dtype=np.int32)
        paths[0, 0, 0] = 0
        paths[1, 0, 0] = 0
        lens = np.ones((2, 1), dtype=np.int32)
        out = np.zeros((2, 2), dtype=np.float32)
        result = _calculate_distance_lca(paths, lens, 2, 1, out)
        self.assertEqual(float(result[0, 1]), 0.0)

    def test_lca_distance_le_terminal_distance(self):
        """Per-pair invariant: LCA distance ≤ terminal-node distance."""
        proximity = DistanceRandomForestProximity()
        proximity.calculate_forest_encoding(estimator=self.model_reg, X=self.X_reg)
        pm, _ = proximity.calculate_distance_matrix(sample_indices=None)

        lca = DistanceRandomForestLCA()
        lca.calculate_forest_encoding(estimator=self.model_reg, X=self.X_reg)
        lm, _ = lca.calculate_distance_matrix(sample_indices=None)

        self.assertTrue(np.all(np.asarray(lm) <= np.asarray(pm) + 1e-6))


class TestTreeHelpers(unittest.TestCase):
    """Direct unit tests for the tree-walking helpers."""

    def setUp(self):
        X, y = make_classification(n_samples=30, n_features=4, random_state=0)
        self.tree = (
            RandomForestClassifier(n_estimators=1, max_depth=3, random_state=0).fit(X, y).estimators_[0].tree_
        )

    def test_compute_parent_array_root_is_minus_one(self):
        parent = _compute_parent_array(self.tree)
        self.assertEqual(parent[0], -1)

    def test_compute_parent_array_each_non_root_has_a_parent(self):
        parent = _compute_parent_array(self.tree)
        for n in range(1, self.tree.node_count):
            self.assertGreaterEqual(parent[n], 0)
            self.assertLess(parent[n], self.tree.node_count)

    def test_compute_node_depths_root_is_zero_and_increases(self):
        depths = _compute_node_depths(self.tree)
        self.assertEqual(depths[0], 0)
        for n in range(self.tree.node_count):
            left = self.tree.children_left[n]
            right = self.tree.children_right[n]
            if left != -1:
                self.assertEqual(depths[left], depths[n] + 1)
            if right != -1:
                self.assertEqual(depths[right], depths[n] + 1)

    def test_build_leaf_to_ancestor_map_predicate_always_true(self):
        parent = _compute_parent_array(self.tree)
        leaf_map = _build_leaf_to_ancestor_map(self.tree, parent, lambda _: True)
        for n in range(self.tree.node_count):
            if self.tree.children_left[n] == -1:
                self.assertEqual(leaf_map[n], n)

    def test_build_leaf_to_ancestor_map_predicate_always_false(self):
        parent = _compute_parent_array(self.tree)
        leaf_map = _build_leaf_to_ancestor_map(self.tree, parent, lambda _: False)
        for n in range(self.tree.node_count):
            if self.tree.children_left[n] == -1:
                self.assertEqual(leaf_map[n], 0)

    def test_validate_mutually_exclusive_single_ok(self):
        _validate_mutually_exclusive(a=1, b=None, c=None)

    def test_validate_mutually_exclusive_two_raises(self):
        with self.assertRaises(ValueError):
            _validate_mutually_exclusive(a=1, b=2, c=None)


class TestDistanceRandomForestBase(unittest.TestCase):
    """Contract tests for the abstract base class."""

    def test_direct_instantiation_raises(self):
        """`DistanceRandomForestBase` is abstract and cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            DistanceRandomForestBase()

    def test_abstract_methods_are_marked(self):
        """Both interface methods must be exposed as abstract on the base class."""
        self.assertTrue(
            getattr(DistanceRandomForestBase.calculate_forest_encoding, "__isabstractmethod__", False)
        )
        self.assertTrue(
            getattr(DistanceRandomForestBase.calculate_distance_matrix, "__isabstractmethod__", False)
        )

    def test_concrete_subclasses_are_instantiable(self):
        """Concrete subclasses implement the abstract methods and instantiate cleanly."""
        DistanceRandomForestProximity()
        DistanceRandomForestLCA()


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
