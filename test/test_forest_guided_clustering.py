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
    plot_forest_guided_clustering,
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
        self.assertEqual(
            first=result.best_k,
            second=2,
            msg="Wrong optimal number of clusters (k)",
        )

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
        self.assertIsNone(result.best_k)

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
        self.assertIsInstance(obj=result.best_k, cls=int)

    def test_forest_guided_feature_importance_output(self):

        cluster_labels = np.random.randint(low=0, high=3, size=self.X.shape[0])

        result = forest_guided_feature_importance(
            X=self.X,
            y=self.y,
            cluster_labels=cluster_labels,
            feature_importance_distance_metric="wasserstein",
        )
        # test output structure
        self.assertIn(member="feature_importance_local", container=result)
        self.assertIn(member="feature_importance_global", container=result)
        self.assertIn(member="data_clustering", container=result)

        # test shape of each output
        self.assertEqual(first=result.feature_importance_local.shape[0], second=self.X.shape[1])
        self.assertEqual(first=result.feature_importance_global.shape[0], second=self.X.shape[1])
        self.assertEqual(first=result.data_clustering.shape[0], second=self.X.shape[0])

    def test_forest_guided_feature_importance_with_y_pred(self):
        rng = np.random.default_rng(seed=0)
        cluster_labels = rng.integers(low=0, high=3, size=self.X.shape[0])
        y_pred = pd.Series(data=self.model.predict(X=self.X))

        result = forest_guided_feature_importance(
            X=self.X,
            y=self.y,
            cluster_labels=cluster_labels,
            y_pred=y_pred,
            feature_importance_distance_metric="wasserstein",
        )
        df = result.data_clustering

        self.assertIn(member="predicted_target", container=df.columns)
        self.assertEqual(first=list(df.columns[:3]), second=["cluster", "target", "predicted_target"])
        # Returned frame is sort_values(...); sort_index() restores rows to original sample order.
        df_sorted = df.sort_index()
        np.testing.assert_array_equal(
            df_sorted["predicted_target"].to_numpy(),
            y_pred.to_numpy(),
        )
        np.testing.assert_array_equal(
            df_sorted["target"].to_numpy(),
            np.asarray(self.y),
        )

    def test_forest_guided_feature_importance_invalid_distance_metric(self):

        cluster_labels = np.random.randint(low=0, high=3, size=self.X.shape[0])

        with self.assertRaises(ValueError):
            forest_guided_feature_importance(
                X=self.X,
                y=self.y,
                cluster_labels=cluster_labels,
                feature_importance_distance_metric="invalid_metric",
            )

    # def test_plot_forest_guided_clustering(self):
    #     result = forest_guided_clustering(
    #         estimator=self.model,
    #         X=self.X,
    #         y=self.y,
    #         clustering_distance_metric=self.distance_metric,
    #         clustering_strategy=self.clustering_strategy,
    #         n_jobs=self.n_jobs,
    #     )

    #     save = os.path.join(self.tmp_path, "test_fgc")

    #     plot_forest_guided_clustering(
    #         ks=result.ks,
    #         scores=result.scores,
    #         mean_ji=result.mean_ji,
    #         cluster_jis=result.cluster_jis,
    #         best_k=result.best_k,
    #         save=save,
    #         show=False,
    #     )
    #     self.assertTrue(
    #         expr=os.path.exists(path=f"{save}_optimizer_results.png"),
    #         msg="Optimizer results plot file was not saved.",
    #     )

    def test_plot_forest_guided_feature_importance(self):
        k = 3
        feature_importance_local = pd.DataFrame(data=np.random.rand(self.X.shape[1], k), index=self.X.columns)
        feature_importance_global = pd.Series(data=np.random.rand(self.X.shape[1]), index=self.X.columns)

        save = os.path.join(self.tmp_path, "test_fgc")

        plot_forest_guided_feature_importance(
            feature_importance_local=feature_importance_local,
            feature_importance_global=feature_importance_global,
            top_n=5,
            num_cols=2,
            save=save,
            show=False,
        )
        self.assertTrue(
            expr=os.path.exists(path=f"{save}_feature_importance.png"),
            msg="Feature importance plot file was not saved.",
        )

    def test_plot_forest_guided_decision_paths(self):

        data_clustering = self.X.copy()
        data_clustering["target"] = self.y
        data_clustering["cluster"] = np.random.randint(low=0, high=3, size=self.X.shape[0])
        data_clustering = data_clustering[["target", "cluster"] + list(self.X.columns)]

        save = os.path.join(self.tmp_path, "test_fgc")

        feature_importance_global = pd.Series(data=np.random.rand(self.X.shape[1]), index=self.X.columns)
        feature_importance_local = pd.DataFrame(data=np.random.rand(self.X.shape[1], 3), index=self.X.columns)

        plot_forest_guided_decision_paths(
            data_clustering=data_clustering,
            feature_importance_global=feature_importance_global,
            feature_importance_local=feature_importance_local,
            model_type=RandomForestClassifier,
            top_n=5,
            draw_distributions=True,
            draw_dotplot=True,
            draw_heatmap=True,
            heatmap_type="static",
            save=save,
            show=False,
        )
        self.assertTrue(
            expr=os.path.exists(path=f"{save}_boxplots.png"),
            msg="Decision path plot file was not saved.",
        )
        self.assertTrue(
            expr=os.path.exists(path=f"{save}_heatmap.png"),
            msg="Decision path plot file was not saved.",
        )
        self.assertTrue(
            expr=os.path.exists(path=f"{save}_dotplot.png"),
            msg="Dotplot file was not saved.",
        )

    def test_color_spec(self):
        result = forest_guided_clustering(
            estimator=self.model,
            X=self.X,
            y=self.y,
            clustering_distance_metric=self.distance_metric,
            clustering_strategy=self.clustering_strategy,
            n_jobs=self.n_jobs,
        )
        results_FI = forest_guided_feature_importance(
            X=self.X,
            y=self.y,
            cluster_labels=result.cluster_labels[result.best_k],
        )

        # No exception; with show=False the API returns (figure, axes).
        out = plot_forest_guided_clustering(
            ks=result.ks,
            scores=result.scores,
            mean_ji=result.mean_ji,
            cluster_jis=result.cluster_jis,
            best_k=result.best_k,
            color_spec={"color_score": "red", "color_ji": "blue"},
            show=False,
        )
        self.assertIsNotNone(out)

        out = plot_forest_guided_feature_importance(
            feature_importance_local=results_FI.feature_importance_local,
            feature_importance_global=results_FI.feature_importance_global,
            top_n=5,
            num_cols=2,
            color_spec={"color_base": "red"},
            show=False,
        )
        self.assertIsNotNone(out)

        out = plot_forest_guided_decision_paths(
            data_clustering=results_FI.data_clustering,
            feature_importance_global=results_FI.feature_importance_global,
            feature_importance_local=results_FI.feature_importance_local,
            model_type=RandomForestClassifier,
            top_n=5,
            draw_distributions=True,
            draw_dotplot=True,
            draw_heatmap=True,
            color_spec={
                "color_target": "Blues",
                "color_target_cat": "Blues",
                "color_features": "viridis",
                "color_features_cat": "Greens",
                "color_NOTUSED": "NOTUSED",
            },
            show=False,
        )
        self.assertIsNotNone(out)
