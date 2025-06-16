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

        self.df = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6], "target": [0, 1, 0]})
        self.X_np = np.array([[1, 4], [2, 5], [3, 6]])
        self.y_np = np.array([0, 1, 0])

    def tearDown(self):
        try:
            shutil.rmtree(self.tmp_path)
        except:
            pass

    def test_check_input_data_with_y_str(self):
        X, y = check_input_data(self.df, "target")
        self.assertEqual(list(X.columns), ["feature_1", "feature_2"])
        self.assertTrue("target" not in X.columns)
        self.assertTrue((y == self.df["target"]).all())

    def test_check_input_data_with_y_array(self):
        X, y = check_input_data(self.X_np, self.y_np)
        np.testing.assert_array_equal(X.values, self.X_np)
        np.testing.assert_array_equal(y.values, self.y_np)

    def test_check_input_estimator_classifier(self):
        model = RandomForestClassifier()
        valid, mtype = check_input_estimator(model)
        self.assertTrue(valid)
        self.assertEqual(mtype, "cla")

    def test_check_input_estimator_regressor(self):
        model = RandomForestRegressor()
        valid, mtype = check_input_estimator(model)
        self.assertTrue(valid)
        self.assertEqual(mtype, "reg")

    def test_check_input_estimator_invalid(self):
        valid, mtype = check_input_estimator("not_a_model")
        self.assertFalse(valid)
        self.assertIsNone(mtype)

    def test_matplotlib_to_plotly_output_format(self):
        colorscale = matplotlib_to_plotly("viridis", pl_entries=10)
        self.assertIsInstance(colorscale, list)
        self.assertEqual(len(colorscale), 10)
        self.assertTrue(all(isinstance(x[1], str) for x in colorscale))
        self.assertTrue(all(x[1].startswith("#") for x in colorscale))

    def test_check_disk_space(self):
        # Should be True for small required bytes
        self.assertTrue(check_disk_space(self.tmp_path, 1024))  # 1 KB
        # Should be False for huge required bytes
        self.assertFalse(check_disk_space(self.tmp_path, 10**15))  # 1 PB

    def test_map_clusters_to_samples_no_mapping(self):
        labels = [0, 1, 0, 2, 1]
        expected = {0: {0, 2}, 1: {1, 4}, 2: {3}}
        result = map_clusters_to_samples(labels)
        self.assertEqual(set(result.keys()), {0, 1, 2})
        self.assertEqual(result, expected)

    def test_map_clusters_to_samples_with_mapping(self):
        labels = [1, 0, 1, 0]
        samples = {i: i + 10 for i in range(len(labels))}
        result = map_clusters_to_samples(labels, samples_mapping=samples)
        self.assertEqual(result[0], {11, 13})
        self.assertEqual(result[1], {10, 12})

    def test_check_k_range(self):
        self.assertEqual(check_k_range(None), (2, 6))
        self.assertEqual(check_k_range(4), (4, 4))
        self.assertEqual(check_k_range((3, 5)), (3, 5))
        self.assertEqual(check_k_range([2, 7]), (2, 7))

    def test_check_k_range_errors(self):
        with self.assertRaises(ValueError):
            check_k_range(1)
        with self.assertRaises(ValueError):
            check_k_range("not valid")
        with self.assertRaises(ValueError):
            check_k_range([2, 3, 4])

    def test_check_sub_sample_size(self):
        self.assertEqual(check_sub_sample_size(None, 5000), 1000)
        self.assertEqual(check_sub_sample_size(0.5, 100), 50)
        self.assertEqual(check_sub_sample_size(1.0, 100), 100)
        self.assertEqual(check_sub_sample_size(20, 100), 20)
        self.assertEqual(check_sub_sample_size(150, 100), 100)
        self.assertEqual(check_sub_sample_size(1, 100), 1)

    def test_check_sub_sample_size_errors(self):
        with self.assertRaises(ValueError):
            check_sub_sample_size(1.1, 100)
        with self.assertRaises(ValueError):
            check_sub_sample_size(0, 100)
        with self.assertRaises(TypeError):
            check_sub_sample_size("ten", 100)
