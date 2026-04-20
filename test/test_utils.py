############################################
# imports
############################################

import os
import shutil
import unittest

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from fgclustering.utils import (
    check_input_data,
    check_input_estimator,
    matplotlib_to_plotly,
    check_disk_space,
    map_clusters_to_samples,
    check_k_range,
    check_sub_sample_size,
)

############################################
# Tests
############################################


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.tmp_path = os.path.join(os.getcwd(), "tmp_fgc")
        Path(self.tmp_path).mkdir(parents=True, exist_ok=True)

        self.df = pd.DataFrame(
            data={"feature_1": [1, 2, 3], "feature_2": [4, 5, 6], "target": [0, 1, 0]}
        )
        self.X_np = np.array([[1, 4], [2, 5], [3, 6]])
        self.y_np = np.array([0, 1, 0])

    def tearDown(self):
        try:
            shutil.rmtree(self.tmp_path)
        except:
            pass

    def test_check_input_data_with_y_str(self):
        X, y, y_pred = check_input_data(X=self.df, y="target")
        self.assertEqual(first=list(X.columns), second=["feature_1", "feature_2"])
        self.assertTrue(expr="target" not in X.columns)
        self.assertTrue(expr=(y == self.df["target"]).all())
        self.assertIsNone(obj=y_pred)

    def test_check_input_data_with_y_array(self):
        X, y, y_pred = check_input_data(X=self.X_np, y=self.y_np)
        np.testing.assert_array_equal(X.values, self.X_np)
        np.testing.assert_array_equal(y.values, self.y_np)
        self.assertIsNone(obj=y_pred)

    def test_check_input_data_with_y_pred(self):
        pred_np = np.array([10, 20, 10])
        _, _, y_pred = check_input_data(X=self.df, y="target", y_pred=pred_np)
        self.assertIsInstance(obj=y_pred, cls=pd.Series)
        np.testing.assert_array_equal(y_pred.to_numpy(), pred_np)
        self.assertEqual(first=y_pred.index.tolist(), second=[0, 1, 2])

        pred_series = pd.Series(data=[1, 0, 1], index=[10, 11, 12])
        _, _, y_pred2 = check_input_data(X=self.df, y="target", y_pred=pred_series)
        np.testing.assert_array_equal(y_pred2.to_numpy(), pred_series.to_numpy())
        self.assertEqual(first=y_pred2.index.tolist(), second=[0, 1, 2])

    def test_check_input_data_with_y_str_invalid(self):
        with self.assertRaises(ValueError):
            check_input_data(X=self.X_np, y="target")

        with self.assertRaises(ValueError):
            check_input_data(X=self.df, y="not_a_column")

    def test_check_input_data_mismatched_lengths(self):
        y_short = np.array([0, 1])
        with self.assertRaises(ValueError):
            check_input_data(X=self.X_np, y=y_short)

        y_pred_short = np.array([0, 1])
        with self.assertRaises(ValueError):
            check_input_data(X=self.X_np, y=self.y_np, y_pred=y_pred_short)

    def test_check_input_estimator_classifier(self):
        model = RandomForestClassifier()
        mtype = check_input_estimator(estimator=model)
        self.assertIs(expr1=mtype, expr2=RandomForestClassifier)

    def test_check_input_estimator_regressor(self):
        model = RandomForestRegressor()
        mtype = check_input_estimator(estimator=model)
        self.assertIs(expr1=mtype, expr2=RandomForestRegressor)

    def test_check_input_estimator_returns_actual_class_for_subclass(self):
        class CustomRF(RandomForestClassifier):
            pass

        model = CustomRF()
        self.assertIs(expr1=check_input_estimator(estimator=model), expr2=CustomRF)

    def test_check_input_estimator_invalid(self):
        mtype = check_input_estimator(estimator="not_a_model")
        self.assertIsNone(obj=mtype)

    def test_matplotlib_to_plotly_output_format(self):
        colorscale = matplotlib_to_plotly(cmap_name="viridis", pl_entries=10)
        self.assertIsInstance(obj=colorscale, cls=list)
        self.assertEqual(first=len(colorscale), second=10)
        self.assertTrue(expr=all(isinstance(x[1], str) for x in colorscale))
        self.assertTrue(expr=all(x[1].startswith("#") for x in colorscale))

    def test_check_disk_space(self):
        # Should be True for small required bytes
        self.assertTrue(expr=check_disk_space(path=self.tmp_path, required_bytes=1024))  # 1 KB
        # Should be False for huge required bytes
        self.assertFalse(expr=check_disk_space(path=self.tmp_path, required_bytes=10**15))  # 1 PB

    def test_map_clusters_to_samples_no_mapping(self):
        labels = [0, 1, 0, 2, 1]
        expected = {0: {0, 2}, 1: {1, 4}, 2: {3}}
        result = map_clusters_to_samples(labels=labels)
        self.assertEqual(first=set(result.keys()), second={0, 1, 2})
        self.assertEqual(first=result, second=expected)

    def test_map_clusters_to_samples_with_mapping(self):
        labels = [1, 0, 1, 0]
        samples = {i: i + 10 for i in range(len(labels))}
        result = map_clusters_to_samples(labels=labels, samples_mapping=samples)
        self.assertEqual(first=result[0], second={11, 13})
        self.assertEqual(first=result[1], second={10, 12})

    def test_check_k_range(self):
        self.assertEqual(first=check_k_range(k=None), second=(2, 6))
        self.assertEqual(first=check_k_range(k=4), second=(4, 4))
        self.assertEqual(first=check_k_range(k=(3, 5)), second=(3, 5))
        self.assertEqual(first=check_k_range(k=[2, 7]), second=(2, 7))

    def test_check_k_range_errors(self):
        with self.assertRaises(ValueError):
            check_k_range(k=1)
        with self.assertRaises(ValueError):
            check_k_range(k="not valid")
        with self.assertRaises(ValueError):
            check_k_range(k=[2, 3, 4])

    def test_check_sub_sample_size(self):
        self.assertEqual(
            first=check_sub_sample_size(
                sub_sample_size=None, n_samples=5000, application="", verbose=0
            ),
            second=1000,
        )
        self.assertEqual(
            first=check_sub_sample_size(sub_sample_size=0.5, n_samples=100, application="", verbose=0),
            second=50,
        )
        self.assertEqual(
            first=check_sub_sample_size(sub_sample_size=1.0, n_samples=100, application="", verbose=0),
            second=100,
        )
        self.assertEqual(
            first=check_sub_sample_size(sub_sample_size=20, n_samples=100, application="", verbose=0),
            second=20,
        )
        self.assertEqual(
            first=check_sub_sample_size(sub_sample_size=150, n_samples=100, application="", verbose=0),
            second=100,
        )
        self.assertEqual(
            first=check_sub_sample_size(sub_sample_size=1, n_samples=100, application="", verbose=0),
            second=1,
        )

    def test_check_sub_sample_size_errors(self):
        with self.assertRaises(ValueError):
            check_sub_sample_size(sub_sample_size=1.1, n_samples=100, application="", verbose=0)
        with self.assertRaises(ValueError):
            check_sub_sample_size(sub_sample_size=0, n_samples=100, application="", verbose=0)
        with self.assertRaises(TypeError):
            check_sub_sample_size(sub_sample_size="ten", n_samples=100, application="", verbose=0)
