############################################
# imports
############################################


import unittest
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification, make_regression

from fgclustering.statistics import FeatureImportance, DistanceJensenShannon, DistanceWasserstein


############################################
# Tests
############################################


class TestFeatureImportance(unittest.TestCase):
    def setUp(self):
        self.n_samples = 100
        self.n_features = 5
        self.clusters = np.repeat([0, 1], self.n_samples // 2)
        self.verbose = 0

    def _generate_classification_data(self):
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=3,
            random_state=42,
        )
        return pd.DataFrame(X, columns=[f"feat_{i}" for i in range(self.n_features)]), pd.Series(y)

    def _generate_regression_data(self):
        X, y = make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=3,
            noise=0.1,
            random_state=1,
        )
        return pd.DataFrame(X, columns=[f"feat_{i}" for i in range(self.n_features)]), pd.Series(y)

    def test_calculate_feature_importance_classification_wasserstein(self):
        X, y = self._generate_classification_data()
        fi = FeatureImportance(distance_metric=DistanceWasserstein(scale_features=True))
        local, global_, df = fi.calculate_feature_importance(
            X, y, self.clusters, model_type="cla", verbose=self.verbose
        )

        self.assertIsInstance(local, pd.DataFrame)
        self.assertIsInstance(global_, pd.Series)
        self.assertIn("cluster", df.columns)
        self.assertIn("target", df.columns)
        self.assertEqual(local.shape[0], X.shape[1])

    def test_calculate_feature_importance_regression_jensenshannon(self):
        X, y = self._generate_regression_data()
        fi = FeatureImportance(distance_metric=DistanceJensenShannon(scale_features=False))
        local, global_, df = fi.calculate_feature_importance(
            X, y, self.clusters, model_type="reg", verbose=self.verbose
        )

        self.assertIsInstance(local, pd.DataFrame)
        self.assertIsInstance(global_, pd.Series)
        self.assertIn("cluster", df.columns)
        self.assertIn("target", df.columns)
        self.assertEqual(local.shape[0], X.shape[1])

    def test_calculate_feature_importance_constant_column(self):
        X, y = self._generate_classification_data()
        X["constant"] = 1  # Add zero-variance feature
        fi = FeatureImportance(distance_metric=DistanceWasserstein(scale_features=True))
        local, global_, _ = fi.calculate_feature_importance(
            X, y, self.clusters, model_type="cla", verbose=self.verbose
        )

        self.assertTrue(local.loc["constant"].isna().all())
        self.assertTrue(np.isnan(global_["constant"]))

    def test_calculate_feature_importance_categorical_feature(self):
        X, y = self._generate_classification_data()
        X["cat_feat"] = np.random.choice(["A", "B", "C"], size=len(X))
        fi = FeatureImportance(distance_metric=DistanceJensenShannon(scale_features=False))
        local, global_, _ = fi.calculate_feature_importance(
            X, y, self.clusters, model_type="cla", verbose=self.verbose
        )

        self.assertIn("cat_feat", local.index)

    def test_calculate_feature_importance_ranking_correctness(self):
        model_type = "cla"  # classifier

        # Create dataset
        X = pd.DataFrame(
            {
                "col_1": [1, 1, 1, 1, 1, 0.9],
                "col_2": [1, 1, 1, 1, 0.9, 0.5],
                "col_3": [1, 1, 1, 0, 0, 1],
                "col_4": [0, 0, 0, 1, 1, 1],
            }
        )
        y = pd.Series([0, 0, 0, 0, 0, 0])
        cluster_labels = np.array([0, 0, 0, 1, 1, 1])

        # Instantiate and compute feature importance
        fi = FeatureImportance(distance_metric=DistanceWasserstein(scale_features=False))
        fi_local, fi_global, X_ranked = fi.calculate_feature_importance(
            X, y, cluster_labels, model_type, verbose=self.verbose
        )

        # Check ordering of features by importance (top = most important)
        ranked_cols = [col for col in X_ranked.columns if col not in ["cluster", "target"]]
        expected_top = ["col_4", "col_3"]

        self.assertEqual(
            ranked_cols[:2],
            expected_top,
            f"Expected top features to be {expected_top}, but got {ranked_cols[:2]}",
        )

    def test_calculate_feature_importance_local_feature_importance(self):
        # Parameters
        thr_distance = 0
        model_type = "cla"

        # Create synthetic test data
        X = pd.DataFrame.from_dict(
            {
                "col_1": [0.9, 1, 0.9, 1, 1, 0.9, 1, 0.9, 1, 0.9, 1, 0.9],
                "col_2": [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 1, 1, 1, 1],
                "col_3": [0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0.1, 0, 0],
                "col_4": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            }
        )
        y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        cluster_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

        # Run feature importance calculation
        fi = FeatureImportance(distance_metric=DistanceWasserstein(scale_features=False))
        fi_local, fi_global, _ = fi.calculate_feature_importance(
            X, y, cluster_labels, model_type, verbose=self.verbose
        )

        # Remove globally unimportant features
        selected_cols = fi_global[fi_global > thr_distance].index
        fi_local = fi_local.loc[selected_cols]

        # Compute median local importance per feature
        median_importance = fi_local.transpose().median()

        # We expect exactly two features to have median local importance > 0.1
        self.assertEqual(
            sum(median_importance > 0.1),
            2,
            f"Expected 2 important features, found {sum(median_importance > 0.1)}",
        )

    def test_calculate_feature_importance_different_distance_funcs(self):
        # Synthetic test data
        X = pd.DataFrame(
            {
                "cont_feature": [0.1, 0.2, 0.2, 0.3, 0.8, 0.9, 0.95, 1.0],  # continuous
                "cat_feature": [0, 0, 0, 0, 1, 1, 1, 1],  # categorical
            }
        )
        y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])  # dummy target
        cluster_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        model_type = "cla"

        # Run for Wasserstein
        fi_w = FeatureImportance(distance_metric=DistanceWasserstein(scale_features=True))
        fi_local_w, fi_global_w, _ = fi_w.calculate_feature_importance(
            X, y, cluster_labels, model_type, verbose=self.verbose
        )

        # Run for Jensen-Shannon
        fi_js = FeatureImportance(distance_metric=DistanceJensenShannon(scale_features=False))
        fi_local_js, fi_global_js, _ = fi_js.calculate_feature_importance(
            X, y, cluster_labels, model_type, verbose=self.verbose
        )

        # Check that the global importance differs
        self.assertFalse(
            fi_global_w.equals(fi_global_js),
            "Global feature importances should differ for Wasserstein vs Jensen-Shannon",
        )

        self.assertNotEqual(
            fi_global_w["cont_feature"],
            fi_global_js["cont_feature"],
            "Continuous feature importance should differ between distance metrics",
        )
        self.assertNotEqual(
            fi_global_w["cat_feature"],
            fi_global_js["cat_feature"],
            "Categorical feature importance should differ between distance metrics",
        )
