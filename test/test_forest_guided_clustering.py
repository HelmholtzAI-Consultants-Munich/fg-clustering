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

from fgclustering.forest_guided_clustering import (
    forest_guided_clustering,
    forest_guided_feature_importance,
    plot_forest_guided_decision_paths,
    plot_forest_guided_feature_importance,
    DistanceRandomForestProximity,
    ClusteringKMedoids,
)


############################################
# Tests
############################################


class TestForestGuidedClustering(unittest.TestCase):

    def setUp(self):
        self.tmp_path = os.path.join(os.getcwd(), "tmp_fgc")
        Path(self.tmp_path).mkdir(parents=True, exist_ok=True)

        # Common test parameters
        self.JI_discart_value = 0.6
        self.JI_bootstrap_iter = 100
        self.random_state = 42
        self.n_jobs = 2

        self.clustering_strategy = ClusteringKMedoids(
            method="pam",
            init="random",
            max_iter=100,
            random_state=self.random_state,
        )

        self.distance_metric = DistanceRandomForestProximity()
        self.distance_metric_memory_eff = DistanceRandomForestProximity(
            memory_efficient=True, dir_distance_matrix=self.tmp_path
        )

        self.X, self.y, self.model = self._train_model()

    def _train_model(self):

        # Generate test data
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=4,
            n_redundant=1,
            n_classes=2,
            n_clusters_per_class=1,
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

    def test_forest_guided_clustering_basic_run(self):
        # Run the forest-guided clustering
        result = forest_guided_clustering(
            estimator=self.model,
            X=self.X,
            y=self.y,
            clustering_distance_metric=self.distance_metric,
            clustering_strategy=self.clustering_strategy,
            n_jobs=self.n_jobs,
        )

        # Check the result
        self.assertEqual(result.k, 2, "Wrong optimal number of clusters (k)")

    def test_forest_guided_clustering_invalid_estimator_type(self):
        with self.assertRaises(ValueError):
            forest_guided_clustering(
                estimator="not_a_model",
                X=self.X,
                y=self.y,
                clustering_distance_metric=self.distance_metric,
                clustering_strategy=self.clustering_strategy,
            )

    def test_forest_guided_clustering_no_stable_clusters(self):
        result = forest_guided_clustering(
            estimator=self.model,
            X=self.X,
            y=self.y,
            clustering_distance_metric=self.distance_metric,
            clustering_strategy=self.clustering_strategy,
            JI_discart_value=1.1,
        )
        self.assertEqual(result.k, 1)

    def test_forest_guided_clustering_memory_efficient(self):
        result = forest_guided_clustering(
            estimator=self.model,
            X=self.X,
            y=self.y,
            clustering_distance_metric=self.distance_metric_memory_eff,
            clustering_strategy=self.clustering_strategy,
            JI_discart_value=self.JI_discart_value,
            JI_bootstrap_iter=self.JI_bootstrap_iter,
            n_jobs=self.n_jobs,
        )
        self.assertIsInstance(result.k, int)

    def test_forest_guided_feature_importance_output(self):

        cluster_labels = np.random.randint(0, 3, size=self.X.shape[0])

        result = forest_guided_feature_importance(
            X=self.X,
            y=self.y,
            cluster_labels=cluster_labels,
            model_type="cla",
            feature_importance_distance_metric="wasserstein",
        )
        # test output structure
        self.assertIn("feature_importance_local", result)
        self.assertIn("feature_importance_global", result)
        self.assertIn("data_clustering", result)

        # test shape of each output
        self.assertEqual(result.feature_importance_local.shape[0], self.X.shape[1])
        self.assertEqual(result.feature_importance_global.shape[0], self.X.shape[1])
        self.assertEqual(result.data_clustering.shape[0], self.X.shape[0])

    def test_forest_guided_feature_importance_invalid_distance_metric(self):

        cluster_labels = np.random.randint(0, 3, size=self.X.shape[0])

        with self.assertRaises(ValueError):
            forest_guided_feature_importance(
                X=self.X,
                y=self.y,
                cluster_labels=cluster_labels,
                model_type="cla",
                feature_importance_distance_metric="invalid_metric",
            )

    def test_plot_forest_guided_feature_importance(self):
        k = 3
        feature_importance_local = pd.DataFrame(np.random.rand(self.X.shape[1], k), index=self.X.columns)
        feature_importance_global = pd.Series(np.random.rand(self.X.shape[1]), index=self.X.columns)

        save = os.path.join(self.tmp_path, "test_fgc")

        plot_forest_guided_feature_importance(
            feature_importance_local,
            feature_importance_global,
            top_n=5,
            num_cols=2,
            save=save,
        )
        self.assertTrue(
            os.path.exists(f"{save}_feature_importance.png"), "Feature importance plot file was not saved."
        )

    def test_plot_forest_guided_decision_paths(self):

        data_clustering = self.X.copy()
        data_clustering["target"] = self.y
        data_clustering["cluster"] = np.random.randint(0, 3, size=self.X.shape[0])
        data_clustering = data_clustering[["target", "cluster"] + list(self.X.columns)]

        save = os.path.join(self.tmp_path, "test_fgc")

        plot_forest_guided_decision_paths(
            data_clustering=data_clustering,
            model_type="cla",
            top_n=5,
            distributions=True,
            heatmap=True,
            heatmap_type="static",
            save=save,
        )
        self.assertTrue(os.path.exists(f"{save}_boxplots.png"), "Decision path plot file was not saved.")
        self.assertTrue(os.path.exists(f"{save}_heatmap.png"), "Decision path plot file was not saved.")
