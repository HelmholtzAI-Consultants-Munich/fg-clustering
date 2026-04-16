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
        self.clusters = np.repeat(a=[0, 1], repeats=self.n_samples // 2)
        self.verbose = 0

    def _generate_classification_data(self):
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=3,
            random_state=42,
        )
        return pd.DataFrame(data=X, columns=[f"feat_{i}" for i in range(self.n_features)]), pd.Series(data=y)

    def _generate_regression_data(self):
        X, y = make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=3,
            noise=0.1,
            random_state=1,
        )
        return pd.DataFrame(data=X, columns=[f"feat_{i}" for i in range(self.n_features)]), pd.Series(data=y)

    def test_calculate_feature_importance_classification_wasserstein(self):
        X, y = self._generate_classification_data()
        fi = FeatureImportance(distance_metric=DistanceWasserstein(scale_features=True))
        local, global_, df = fi.calculate_feature_importance(
            X=X, y=y, y_pred=None, cluster_labels=self.clusters, verbose=self.verbose
        )

        self.assertIsInstance(obj=local, cls=pd.DataFrame)
        self.assertIsInstance(obj=global_, cls=pd.Series)
        self.assertIn(member="cluster", container=df.columns)
        self.assertIn(member="target", container=df.columns)
        self.assertNotIn(member="predicted_target", container=df.columns)
        self.assertEqual(first=local.shape[0], second=X.shape[1])

    def test_calculate_feature_importance_regression_jensenshannon(self):
        X, y = self._generate_regression_data()
        fi = FeatureImportance(distance_metric=DistanceJensenShannon(scale_features=False))
        local, global_, df = fi.calculate_feature_importance(
            X=X, y=y, y_pred=None, cluster_labels=self.clusters, verbose=self.verbose
        )

        self.assertIsInstance(obj=local, cls=pd.DataFrame)
        self.assertIsInstance(obj=global_, cls=pd.Series)
        self.assertIn(member="cluster", container=df.columns)
        self.assertIn(member="target", container=df.columns)
        self.assertNotIn(member="predicted_target", container=df.columns)
        self.assertEqual(first=local.shape[0], second=X.shape[1])

    def test_calculate_feature_importance_y_pred_stored_in_output(self):
        X, y = self._generate_classification_data()
        y_pred = pd.Series(data=(y + 1) % 2, index=y.index)
        fi = FeatureImportance(distance_metric=DistanceWasserstein(scale_features=True))
        _, _, df = fi.calculate_feature_importance(
            X=X, y=y, y_pred=y_pred, cluster_labels=self.clusters, verbose=self.verbose
        )

        self.assertIn(member="predicted_target", container=df.columns)
        self.assertEqual(first=list(df.columns[:3]), second=["cluster", "target", "predicted_target"])
        # Returned frame is sort_values(...); sort_index() restores rows to original sample order.
        df_sorted = df.sort_index()
        np.testing.assert_array_equal(df_sorted["predicted_target"].to_numpy(), y_pred.to_numpy())
        np.testing.assert_array_equal(df_sorted["target"].to_numpy(), y.to_numpy())

    def test_calculate_feature_importance_constant_column(self):
        X, y = self._generate_classification_data()
        X["constant"] = 1  # Add zero-variance feature
        fi = FeatureImportance(distance_metric=DistanceWasserstein(scale_features=True))
        local, global_, _ = fi.calculate_feature_importance(
            X=X, y=y, y_pred=None, cluster_labels=self.clusters, verbose=self.verbose
        )

        self.assertTrue(expr=local.loc["constant"].isna().all())
        self.assertTrue(expr=np.isnan(global_["constant"]))

    def test_calculate_feature_importance_categorical_feature(self):
        X, y = self._generate_classification_data()
        X["cat_feat"] = np.random.choice(a=["A", "B", "C"], size=len(X))
        fi = FeatureImportance(distance_metric=DistanceJensenShannon(scale_features=False))
        local, global_, _ = fi.calculate_feature_importance(
            X=X, y=y, y_pred=None, cluster_labels=self.clusters, verbose=self.verbose
        )

        self.assertIn(member="cat_feat", container=local.index)

    def test_calculate_feature_importance_ranking_correctness(self):
        # Create dataset
        X = pd.DataFrame(
            data={
                "col_1": [1, 1, 1, 1, 1, 0.9],
                "col_2": [1, 1, 1, 1, 0.9, 0.5],
                "col_3": [1, 1, 1, 0, 0, 1],
                "col_4": [0, 0, 0, 1, 1, 1],
            }
        )
        y = pd.Series(data=[0, 0, 0, 0, 0, 0])
        cluster_labels = np.array([0, 0, 0, 1, 1, 1])

        # Instantiate and compute feature importance
        fi = FeatureImportance(distance_metric=DistanceWasserstein(scale_features=False))
        fi_local, fi_global, X_ranked = fi.calculate_feature_importance(
            X=X, y=y, y_pred=None, cluster_labels=cluster_labels, verbose=self.verbose
        )

        # Check ordering of features by importance (top = most important)
        ranked_cols = [col for col in X_ranked.columns if col not in ["cluster", "target"]]
        expected_top = ["col_4", "col_3"]

        self.assertEqual(
            first=ranked_cols[:2],
            second=expected_top,
            msg=f"Expected top features to be {expected_top}, but got {ranked_cols[:2]}",
        )

    def test_calculate_feature_importance_local_feature_importance(self):
        # Parameters
        thr_distance = 0

        # Create synthetic test data
        X = pd.DataFrame.from_dict(
            data={
                "col_1": [0.9, 1, 0.9, 1, 1, 0.9, 1, 0.9, 1, 0.9, 1, 0.9],
                "col_2": [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 1, 1, 1, 1],
                "col_3": [0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0.1, 0, 0],
                "col_4": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            }
        )
        y = pd.Series(data=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        cluster_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

        # Run feature importance calculation
        fi = FeatureImportance(distance_metric=DistanceWasserstein(scale_features=False))
        fi_local, fi_global, _ = fi.calculate_feature_importance(
            X=X, y=y, y_pred=None, cluster_labels=cluster_labels, verbose=self.verbose
        )

        # Remove globally unimportant features
        selected_cols = fi_global[fi_global > thr_distance].index
        fi_local = fi_local.loc[selected_cols]

        # Compute median local importance per feature
        median_importance = fi_local.transpose().median()

        # We expect exactly two features to have median local importance > 0.1
        self.assertEqual(
            first=sum(median_importance > 0.1),
            second=2,
            msg=f"Expected 2 important features, found {sum(median_importance > 0.1)}",
        )

    def test_calculate_feature_importance_different_distance_funcs(self):
        # Synthetic test data
        X = pd.DataFrame(
            data={
                "cont_feature": [0.1, 0.2, 0.2, 0.3, 0.8, 0.9, 0.95, 1.0],  # continuous
                "cat_feature": [0, 0, 0, 0, 1, 1, 1, 1],  # categorical
            }
        )
        y = pd.Series(data=[0, 0, 0, 0, 1, 1, 1, 1])  # dummy target
        cluster_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        # Run for Wasserstein
        fi_w = FeatureImportance(distance_metric=DistanceWasserstein(scale_features=True))
        fi_local_w, fi_global_w, _ = fi_w.calculate_feature_importance(
            X=X, y=y, y_pred=None, cluster_labels=cluster_labels, verbose=self.verbose
        )

        # Run for Jensen-Shannon
        fi_js = FeatureImportance(distance_metric=DistanceJensenShannon(scale_features=False))
        fi_local_js, fi_global_js, _ = fi_js.calculate_feature_importance(
            X=X, y=y, y_pred=None, cluster_labels=cluster_labels, verbose=self.verbose
        )

        # Check that the global importance differs
        self.assertFalse(
            expr=fi_global_w.equals(other=fi_global_js),
            msg="Global feature importances should differ for Wasserstein vs Jensen-Shannon",
        )

        self.assertNotEqual(
            first=fi_global_w["cont_feature"],
            second=fi_global_js["cont_feature"],
            msg="Continuous feature importance should differ between distance metrics",
        )
        self.assertNotEqual(
            first=fi_global_w["cat_feature"],
            second=fi_global_js["cat_feature"],
            msg="Categorical feature importance should differ between distance metrics",
        )
