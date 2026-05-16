############################################
# imports
############################################

import unittest
import numpy as np
import pandas as pd
import shutil
import os

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_blobs, make_regression

from fgclustering.distance import (
    DistanceRandomForestLCA,
    DistanceRandomForestProximity,
    _assign_labels_proximity,
    _calculate_inertia_proximity,
)
from fgclustering.clustering import ClusteringKMedoids, ClusteringClara

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


class TestClusteringKMedoids(unittest.TestCase):
    def setUp(self):

        self.tmp_path = os.path.join(os.getcwd(), "tmp_fgc")
        Path(self.tmp_path).mkdir(parents=True, exist_ok=True)

        self.random_state = 42

        self.X, self.y, self.model = self._train_model()
        self.sample_indices = np.arange(stop=100)

    def _train_model(self):

        # Generate test data
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
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

    def test_run_clustering_output_shape(self):
        distance = DistanceRandomForestProximity(memory_efficient=False)
        distance.calculate_forest_encoding(estimator=self.model, X=self.X)

        clustering = ClusteringKMedoids()
        labels = clustering.run_clustering(
            k=3,
            distance_metric=distance,
            sample_indices=self.sample_indices,
            random_state_subsampling=None,
            verbose=1,
        )

        self.assertEqual(first=len(labels), second=len(self.sample_indices))
        self.assertTrue(expr=np.all(a=np.isin(element=labels, test_elements=[1, 2, 3])))

    def test_run_clustering_memory_efficient(self):
        distance = DistanceRandomForestProximity(memory_efficient=True, dir_distance_matrix=self.tmp_path)
        distance.calculate_forest_encoding(estimator=self.model, X=self.X)

        clustering = ClusteringKMedoids()
        labels = clustering.run_clustering(
            k=4,
            distance_metric=distance,
            sample_indices=self.sample_indices,
            random_state_subsampling=None,
            verbose=1,
        )

        self.assertEqual(first=len(labels), second=len(self.sample_indices))
        self.assertTrue(expr=np.all(a=np.isin(element=labels, test_elements=[1, 2, 3, 4])))

    def test_run_clustering_invalid_sample_indices(self):
        distance = DistanceRandomForestProximity(memory_efficient=False)
        distance.calculate_forest_encoding(estimator=self.model, X=self.X)

        clustering = ClusteringKMedoids()
        with self.assertRaises(IndexError):
            clustering.run_clustering(
                k=3,
                distance_metric=distance,
                sample_indices=np.array([999, 1000]),
                random_state_subsampling=None,
                verbose=1,
            )

    def test_run_clustering_different_k_values(self):
        distance = DistanceRandomForestProximity(memory_efficient=False)
        distance.calculate_forest_encoding(estimator=self.model, X=self.X)

        clustering = ClusteringKMedoids()
        for k in [2, 5, 7]:
            labels = clustering.run_clustering(
                k=k,
                distance_metric=distance,
                sample_indices=self.sample_indices,
                random_state_subsampling=None,
                verbose=1,
            )
            self.assertEqual(first=len(labels), second=len(self.sample_indices))
            self.assertEqual(first=len(np.unique(ar=labels)), second=k)


class TestClusteringClara(unittest.TestCase):
    def setUp(self):

        self.tmp_path = os.path.join(os.getcwd(), "tmp_fgc")
        Path(self.tmp_path).mkdir(parents=True, exist_ok=True)

        self.random_state = 42

        self.X, self.y, self.model = self._train_model()
        self.sample_indices = np.arange(stop=100)

    def _train_model(self):

        # Generate test data
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
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

    def test_run_clustering_output_shape(self):
        distance = DistanceRandomForestProximity(memory_efficient=False)
        distance.calculate_forest_encoding(estimator=self.model, X=self.X)

        clara = ClusteringClara(sub_sample_size=50, sampling_iter=5)
        labels = clara.run_clustering(
            k=3,
            distance_metric=distance,
            sample_indices=self.sample_indices,
            random_state_subsampling=None,
            verbose=1,
        )

        self.assertEqual(first=len(labels), second=len(self.sample_indices))
        self.assertEqual(first=len(np.unique(ar=labels)), second=3)

    def test_run_clustering_memory_efficient(self):
        distance = DistanceRandomForestProximity(memory_efficient=True, dir_distance_matrix=self.tmp_path)
        distance.calculate_forest_encoding(estimator=self.model, X=self.X)

        clara = ClusteringClara(sub_sample_size=50, sampling_iter=5)
        labels = clara.run_clustering(
            k=4,
            distance_metric=distance,
            sample_indices=self.sample_indices,
            random_state_subsampling=None,
            verbose=1,
        )

        self.assertEqual(first=len(labels), second=len(self.sample_indices))
        self.assertEqual(first=len(np.unique(ar=labels)), second=4)

    def test_run_clustering_same_seed_gives_same_result(self):
        distance = DistanceRandomForestProximity(memory_efficient=False)
        distance.calculate_forest_encoding(estimator=self.model, X=self.X)

        clara1 = ClusteringClara(sub_sample_size=50, sampling_iter=5, random_state=0)
        clara2 = ClusteringClara(sub_sample_size=50, sampling_iter=5, random_state=0)

        labels1 = clara1.run_clustering(
            k=3,
            distance_metric=distance,
            sample_indices=self.sample_indices,
            random_state_subsampling=None,
            verbose=1,
        )
        labels2 = clara2.run_clustering(
            k=3,
            distance_metric=distance,
            sample_indices=self.sample_indices,
            random_state_subsampling=None,
            verbose=1,
        )

        np.testing.assert_array_equal(labels1, labels2)

    def test_run_clustering_auto_iteration_fallback(self):
        distance = DistanceRandomForestProximity(memory_efficient=False)
        distance.calculate_forest_encoding(estimator=self.model, X=self.X)

        clara = ClusteringClara(sub_sample_size=50, sampling_iter=None)
        labels = clara.run_clustering(
            k=3,
            distance_metric=distance,
            sample_indices=self.sample_indices,
            random_state_subsampling=None,
            verbose=1,
        )

        self.assertEqual(first=len(labels), second=len(self.sample_indices))

    def test_run_clustering_subsample_size_as_fraction(self):
        distance = DistanceRandomForestProximity(memory_efficient=False)
        distance.calculate_forest_encoding(estimator=self.model, X=self.X)

        clara = ClusteringClara(sub_sample_size=0.5, sampling_iter=3)
        labels = clara.run_clustering(
            k=2,
            distance_metric=distance,
            sample_indices=self.sample_indices,
            random_state_subsampling=None,
            verbose=1,
        )

        self.assertEqual(first=len(labels), second=len(self.sample_indices))

    def test_run_clustering_clara_improves_with_more_iterations(self):
        # Create 3 clear Gaussian clusters
        X, y_true = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=42)
        X = pd.DataFrame(data=X)

        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X=X, y=y_true)

        sample_indices = np.arange(stop=len(X))
        distance = DistanceRandomForestProximity(memory_efficient=False)
        distance.calculate_forest_encoding(estimator=model, X=X)

        # Clara with few iterations
        clara_few = ClusteringClara(sub_sample_size=0.5, sampling_iter=1, random_state=42)
        labels_few = clara_few.run_clustering(
            k=3,
            distance_metric=distance,
            sample_indices=sample_indices,
            random_state_subsampling=None,
            verbose=1,
        )
        inertia_few = _calculate_inertia_proximity(
            terminals=distance.terminals,
            sample_idx=sample_indices,
            medoids_idx=sample_indices[np.unique(ar=labels_few, return_index=True)[1]],
        )

        # Clara with more iterations
        clara_many = ClusteringClara(sub_sample_size=0.5, sampling_iter=10, random_state=42)
        labels_many = clara_many.run_clustering(
            k=3,
            distance_metric=distance,
            sample_indices=sample_indices,
            random_state_subsampling=None,
            verbose=1,
        )
        inertia_many = _calculate_inertia_proximity(
            terminals=distance.terminals,
            sample_idx=sample_indices,
            medoids_idx=sample_indices[np.unique(ar=labels_many, return_index=True)[1]],
        )

        # The inertia from more iterations should not be worse
        assert (
            inertia_many <= inertia_few + 1e-5
        ), "CLARA with more iterations did not yield better or equal clustering"

    def test_calculate_inertia_basic(self):
        # Simulate simple terminal matrix (n_samples x n_estimators)
        terminals = np.array(
            [
                [1, 1, 2],  # sample 0
                [1, 1, 2],  # sample 1 (identical to 0)
                [3, 3, 3],  # sample 2 (distinct)
            ],
            dtype=np.int32,
        )

        sample_idx = np.array([0, 1, 2])
        medoids_idx = np.array([0])  # only one medoid identical to sample 0 and 1

        inertia = _calculate_inertia_proximity(
            terminals=terminals, sample_idx=sample_idx, medoids_idx=medoids_idx
        )

        # Distance for samples 0 and 1 should be 0
        # Distance for sample 2 should be 1.0 (no match)
        expected = 0 + 0 + 1.0
        assert np.isclose(a=inertia, b=expected), f"Expected inertia {expected}, got {inertia}"

    def test_asign_labels_basic(self):
        terminals = np.array(
            [
                [1, 1, 2],  # sample 0
                [1, 1, 2],  # sample 1
                [3, 3, 3],  # sample 2
            ],
            dtype=np.int32,
        )

        sample_idx = np.array([0, 1, 2])
        medoids_idx = np.array([0, 2])  # Two medoids: sample 0 and sample 2

        labels = _assign_labels_proximity(terminals=terminals, sample_idx=sample_idx, medoids_idx=medoids_idx)

        # Expect:
        # sample 0 -> medoid 0 (label 0)
        # sample 1 -> medoid 0 (label 0)
        # sample 2 -> medoid 1 (label 1)
        expected = np.array([0, 0, 1], dtype=np.int16)

        assert np.array_equal(a1=labels, a2=expected), f"Expected labels {expected}, got {labels}"

    def test_clara_with_lca_distance_uses_lca_inertia(self):
        """CLARA + DistanceRandomForestLCA must complete and produce labels for every sample."""
        X, _, model = _build_regression_forest()
        dist = DistanceRandomForestLCA()
        dist.calculate_forest_encoding(estimator=model, X=X)
        clara = ClusteringClara(sub_sample_size=20, sampling_iter=3, random_state=0)
        labels = clara.run_clustering(
            k=2,
            distance_metric=dist,
            sample_indices=np.arange(len(X)),
            random_state_subsampling=0,
            verbose=0,
        )

        self.assertEqual(labels.shape, (len(X),))
        self.assertTrue(set(np.unique(labels)).issubset({1, 2}))

    def test_compute_inertia_consistency_with_distance_matrix(self):
        """`compute_inertia` must equal `sum_i min_m distance_matrix[i, m]`."""
        for cls in (DistanceRandomForestProximity, DistanceRandomForestLCA):
            with self.subTest(distance_class=cls.__name__):
                X, _, model = _build_regression_forest()
                dist = cls()
                dist.calculate_forest_encoding(estimator=model, X=X)
                sample_idx = np.arange(len(X))
                medoids_idx = np.array([0, 5, 10, 15])
                inertia = dist.compute_inertia(sample_idx, medoids_idx)
                matrix, file = dist.calculate_distance_matrix(sample_indices=None)
                expected = float(np.asarray(matrix)[:, medoids_idx].min(axis=1).sum())
                self.assertAlmostEqual(float(inertia), expected, places=4)
                dist.remove_distance_matrix(matrix, file)
