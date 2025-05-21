############################################
# imports
############################################


import numpy as np
import pandas as pd
from fgclustering.utils import *
from fgclustering.statistics import (
    compute_balanced_average_impurity,
    compute_total_within_cluster_variation,
    calculate_feature_importance,
)


############################################
# Tests
############################################


def test_compute_balanced_average_impurity():
    # test data
    categorical_values = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    cluster_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    result = compute_balanced_average_impurity(categorical_values, cluster_labels)

    assert result == 0.0, "error: impurity should be 0"


def test_compute_total_within_cluster_variation():
    # test data
    continuous_values = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    cluster_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    result = compute_total_within_cluster_variation(continuous_values, cluster_labels)

    assert result == 0.0, "error: within cluster variation should be 0"


def test_calculate_global_feature_importance():
    # test if anova test filters out features 1 and 2 which are the same in both clusters and
    # leaves features 3 and 4 which are clearly different in both clusters
    # parameters
    model_type = "classifier"

    # test data
    X = pd.DataFrame.from_dict(
        {
            "col_1": [1, 1, 1, 1, 1, 0.9],
            "col_2": [1, 1, 1, 1, 0.9, 0.5],
            "col_3": [1, 1, 1, 0, 0, 1],
            "col_4": [0, 0, 0, 1, 1, 1],
        }
    )
    y = pd.Series([0, 0, 0, 0, 0, 0])
    cluster_labels = np.array([0, 0, 0, 1, 1, 1])

    feature_importance_local, feature_importance_global, X_ranked = calculate_feature_importance(X, y, cluster_labels, model_type=model_type)

    X_ranked.drop("cluster", axis=1, inplace=True)
    assert list(X_ranked.columns) == [
        "target",
        "col_4",
        "col_3",
        "col_2",
        "col_1",
    ], "error: global feature importance returns wrong ordering"


def test_calculate_local_feature_importance():
    # test if clusterwise importance is high for feature 2 and 4 and low for feature 1 and 3
    # parameters
    thr_distance = 0
    model_type = "classifier"

    # test data
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

    feature_importance_local, feature_importance_global, X_ranked = calculate_feature_importance(X, y, cluster_labels, model_type=model_type)
    for column in X.columns:
        if feature_importance_global.loc[column] > thr_distance:
            X.drop(column, axis=1, inplace=True)

    importance = feature_importance_local
    result = importance.transpose().median()
    
    assert sum(result > 0.1) == 2, "error: wrong number of features with highest feature importance"


def test_feature_importance_all_below_threshold():
    X = pd.DataFrame(
        {
            "col_1": [1, 1, 1, 1],
            "col_2": [0, 0, 0, 0],
        }
    )
    y = pd.Series([0, 0, 1, 1])
    cluster_labels = np.array([0, 0, 1, 1])
    model_type = "classifier"

    feature_importance_local, feature_importance_global, X_ranked = calculate_feature_importance(X, y, cluster_labels, model_type=model_type)

    # Here we assume a threshold of 0.5, which no feature should exceed
    thr_distance = 0.5
    retained = feature_importance_global[feature_importance_global > thr_distance]
    
    assert retained.empty, "error: no features should pass the threshold"


def test_feature_importance_with_different_distance_funcs():
    # Construct data with clearly distinguishable distributions
    X = pd.DataFrame({
        "cont_feature": [0.1, 0.2, 0.2, 0.3, 0.8, 0.9, 0.95, 1.0],  # continuous
        "cat_feature": [0, 0, 0, 0, 1, 1, 1, 1],                    # categorical
    })
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])  # Dummy target
    cluster_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    fi_local_w, fi_global_w, _ = calculate_feature_importance(
        X, y, cluster_labels, distance_func="wasserstein", model_type="classifier"
    )

    fi_local_js, fi_global_js, _ = calculate_feature_importance(
        X, y, cluster_labels, distance_func="jensen-shannon", model_type="classifier"
    )

    # They should not be exactly the same
    assert not fi_global_w.equals(fi_global_js), "error: distance functions return identical results"

