############################################
# Imports
############################################

import numpy as np
import pandas as pd
from typing import Union
from sklearn.utils import Bunch

from .utils import check_input_data, check_input_estimator, check_sub_sample_size, check_k_range
from .clustering import ClusteringKMedoids, ClusteringClara
from .distance import DistanceWasserstein, DistanceJensenShannon, DistanceRandomForestProximity
from .optimizer import Optimizer
from .statistics import FeatureImportance
from .plotting import (
    _plot_feature_importance,
    _plot_distributions,
    _plot_heatmap_regression,
    _plot_heatmap_classification,
)


############################################
# Forest-guided Clustering API
############################################


def forest_guided_clustering(
    estimator,
    X,
    y,
    clustering_distance_metric: DistanceRandomForestProximity,
    clustering_strategy: Union[ClusteringKMedoids, ClusteringClara],
    k: int = None,
    JI_bootstrap_iter: int = 100,
    JI_bootstrap_sample_size: int = None,
    JI_discart_value: int = 0.6,
    n_jobs: int = 1,
    random_state: int = 42,
    verbose: int = 1,
):
    np.random.seed(random_state)

    # check estimator class
    valid_estimator, model_type = check_input_estimator(estimator)
    if not valid_estimator:
        raise ValueError("Model must be a scikit-learn RandomForestClassifier or RandomForestRegressor")

    # format input data
    X, y = check_input_data(X, y)

    # format range of k values
    k_range = check_k_range(k=k)

    # check bootstrap sample size
    JI_bootstrap_sample_size = check_sub_sample_size(
        sub_sample_size=JI_bootstrap_sample_size, n_samples=len(y), verbose=verbose
    )

    # optimize k
    clustering_distance_metric.calculate_terminals(estimator=estimator, X=X)

    optimizer = Optimizer(
        distance_metric=clustering_distance_metric,
        clustering=clustering_strategy,
        verbose=verbose,
        random_state=random_state,
    )
    k, cluster_score, cluster_stability, cluster_labels = optimizer.optimizeK(
        y=y,
        k_range=k_range,
        JI_bootstrap_iter=JI_bootstrap_iter,
        JI_bootstrap_sample_size=JI_bootstrap_sample_size,
        JI_discart_value=JI_discart_value,
        model_type=model_type,
        n_jobs=n_jobs,
    )

    return Bunch(
        k=k,
        cluster_score=cluster_score,
        cluster_stability=cluster_stability,
        cluster_labels=cluster_labels,
        model_type=model_type,
    )


def forest_guided_feature_importance(
    X,
    y,
    cluster_labels,
    model_type,
    feature_importance_distance_metric="wasserstein",
    verbose: int = 1,
):
    X, y = check_input_data(X, y)

    if feature_importance_distance_metric == "wasserstein":
        feature_importance_distance_metric = DistanceWasserstein(scale_features=True)
    elif feature_importance_distance_metric == "jensenshannon":
        feature_importance_distance_metric = DistanceJensenShannon(scale_features=False)
    else:
        raise ValueError(f"Distance metric {feature_importance_distance_metric} is not available!")

    feature_importance = FeatureImportance(
        distance_metric=feature_importance_distance_metric, verbose=verbose
    )

    fi_local, fi_global, data_clustering = feature_importance.calculate_feature_importance(
        X=X, y=y, cluster_labels=cluster_labels, model_type=model_type
    )

    return Bunch(
        feature_importance_local=fi_local,
        feature_importance_global=fi_global,
        data_clustering=data_clustering,
    )


def plot_forest_guided_feature_importance(
    feature_importance_local,
    feature_importance_global,
    top_n: int = None,
    num_cols: int = 4,
    save: str = None,
):

    # select top n features for plotting
    assert isinstance(feature_importance_global, pd.Series), (
        f"Expected `feature_importance_global` to be a Series, but got {type(feature_importance_global)} "
        f"with shape {getattr(feature_importance_global, 'shape', 'N/A')}."
    )

    selected_features = feature_importance_global.sort_values(ascending=False).index.tolist()
    if top_n:
        selected_features = selected_features[:top_n]

    _plot_feature_importance(
        feature_importance_global=feature_importance_global[selected_features],
        feature_importance_local=feature_importance_local.loc[selected_features, :],
        top_n=top_n,
        num_cols=num_cols,
        save=save,
    )


def plot_forest_guided_decision_paths(
    data_clustering: pd.DataFrame,
    model_type: str,
    distributions: bool = True,
    heatmap: bool = True,
    heatmap_type: str = "static",
    top_n: int = None,
    num_cols: int = 6,
    cmap_target_dict: dict = None,
    save: str = None,
):
    # select top n features and cluster, target for plotting
    if top_n:
        data_clustering_selected_featues = data_clustering.iloc[:, : top_n + 2]
    else:
        data_clustering_selected_featues = data_clustering

    if distributions:
        _plot_distributions(data_clustering_selected_featues, top_n, num_cols, cmap_target_dict, save)

    if heatmap:
        if model_type == "reg":
            _plot_heatmap_regression(data_clustering_selected_featues, top_n, heatmap_type, save)
        elif model_type == "cla":
            _plot_heatmap_classification(
                data_clustering_selected_featues, top_n, heatmap_type, cmap_target_dict, save
            )
