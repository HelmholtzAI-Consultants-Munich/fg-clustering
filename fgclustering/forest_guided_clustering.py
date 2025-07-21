############################################
# Imports
############################################

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from sklearn.utils import Bunch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


from .utils import check_input_data, check_input_estimator, check_sub_sample_size, check_k_range
from .clustering import ClusteringKMedoids, ClusteringClara
from .distance import DistanceWasserstein, DistanceJensenShannon, DistanceRandomForestProximity
from .optimizer import Optimizer
from .statistics import FeatureImportance
from .plotting import (
    plot_feature_importance,
    plot_distributions,
    plot_heatmap_regression,
    plot_heatmap_classification,
)


############################################
# Forest-guided Clustering API
############################################


def forest_guided_clustering(
    estimator: Union[RandomForestClassifier, RandomForestRegressor],
    X: pd.DataFrame,
    y: Union[str, pd.Series],
    clustering_distance_metric: DistanceRandomForestProximity,
    clustering_strategy: Union[ClusteringKMedoids, ClusteringClara],
    k: Optional[Union[int, Tuple[int, int]]] = None,
    JI_bootstrap_iter: Optional[int] = 100,
    JI_bootstrap_sample_size: Optional[Union[int, float]] = None,
    JI_discart_value: Optional[float] = 0.6,
    n_jobs: Optional[int] = 1,
    random_state: Optional[int] = 42,
    verbose: Optional[int] = 1,
) -> Bunch:
    """
    Perform forest-guided clustering using proximity information derived from a Random Forest model.

    This function uses the structure of a trained Random Forest to extract proximity-based similarities
    between instances. Based on the proximity (number of shared terminal nodes), a distance matrix is calculated,
    which is used to perform clustering via K-Medoids or CLARA. The number of clusters (k) can be optimized
    using internal stability metrics such as the Jaccard Index (JI) across multiple bootstraps. The
    optimization prioritizes both cluster stability and model-specific scoring (e.g., balanced impurity for
    classification, within-cluster variance for regression). The function returns the best k with its score,
    stability and assigned cluster labels for each sample of the input data.

    :param estimator: A fitted RandomForestClassifier or RandomForestRegressor model.
    :type estimator: Union[RandomForestClassifier, RandomForestRegressor]
    :param X: Input feature matrix.
    :type X: pandas.DataFrame
    :param y: Target variable, i.e. target values or name of the target column in X.
    :type y: Union[str, pandas.Series]
    :param clustering_distance_metric: An instance of DistanceRandomForestProximity.
    :type clustering_distance_metric: DistanceRandomForestProximity
    :param clustering_strategy: An instance of ClusteringKMedoids or ClusteringClara.
    :type clustering_strategy: Union[ClusteringKMedoids, ClusteringClara]
    :param k: If int, number of clusters is fixed. If tuple, number of clusters is optimized within the given range. If None number of clusters is optimized within the range of 2 to 6. Default: None
    :type k: Optional[Union[int, Tuple[int, int]]]
    :param JI_bootstrap_iter: Number of bootstrap iterations for Jaccard Index evaluation. Default: 100
    :type JI_bootstrap_iter: Optional[int]
    :param JI_bootstrap_sample_size: Number or proportion of samples to draw for each JI bootstrap. If None, computes an adaptive subsample ratio based on dataset size, constrained between 10% and 80%, targeting approximately 1,000 samples. Default: None.
    :type JI_bootstrap_sample_size: Optional[Union[int, float]]
    :param JI_discart_value: Jaccard Index threshold to discard unstable clusters. Default: 0.6
    :type JI_discart_value: Optional[float]
    :param n_jobs: Number of parallel jobs. Default: 1.
    :type n_jobs: Optional[int]
    :param random_state: Random seed for reproducibility. Default: 42.
    :type random_state: Optional[int]
    :param verbose: Verbosity level (0 = silent, 1 = progress messages). Default: 1.
    :type verbose: Optional[int]

    :return: A Bunch object containing k, cluster scores, cluster stability, labels, and model type.
    :rtype: Bunch
    """
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
        sub_sample_size=JI_bootstrap_sample_size,
        n_samples=len(y),
        application="Jaccard Index computation",
        verbose=verbose,
    )

    # optimize k
    clustering_distance_metric.calculate_terminals(estimator=estimator, X=X)

    optimizer = Optimizer(
        distance_metric=clustering_distance_metric,
        clustering_strategy=clustering_strategy,
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
        verbose=verbose,
    )

    return Bunch(
        k=k,
        cluster_score=cluster_score,
        cluster_stability=cluster_stability,
        cluster_labels=cluster_labels,
        model_type=model_type,
    )


def forest_guided_feature_importance(
    X: pd.DataFrame,
    y: Union[str, pd.Series],
    cluster_labels: np.ndarray,
    model_type: str,
    feature_importance_distance_metric: Optional[str] = "wasserstein",
    verbose: Optional[int] = 1,
) -> Bunch:
    """
    Compute forest-guided feature importance by quantifying how much each feature contributes to cluster separation.

    This function compares the distribution of each feature within a cluster to its distribution across the entire dataset,
    using either the Wasserstein or Jensen-Shannon distance metric. Wasserstein distance is recommended for a dataset with primarily
    continuous features, while Jensen-Shannon is better suited for categorical features. The resulting scores indicate
    which features are most important for distinguishing clusters, providing interpretability into the model’s decision structure.

    :param X: Input feature matrix.
    :type X: pandas.DataFrame
    :param y: Target variable, i.e. target values or name of the target column in X.
    :type y: Union[str, pandas.Series]
    :param cluster_labels: Labels from forest-guided clustering. Output of `forest_guided_clustering()`.
    :type cluster_labels: numpy.ndarray
    :param model_type: Type of model, either "cla" for classification or "reg" for regression. Output of `forest_guided_clustering()`.
    :type model_type: str
    :param feature_importance_distance_metric: Distance metric for computing feature importance ("wasserstein" or "jensenshannon"). Default: "wasserstein".
    :type feature_importance_distance_metric: Optional[str]
    :param verbose: Verbosity level (0 = silent, 1 = progress messages). Default: 1.
    :type verbose: Optional[int]

    :return: A Bunch object containing local feature importances, global feature importances and the clustering data ordered by the global feature importance.
    :rtype: Bunch
    """
    X, y = check_input_data(X, y)

    if feature_importance_distance_metric == "wasserstein":
        feature_importance_distance_metric = DistanceWasserstein(scale_features=True)
    elif feature_importance_distance_metric == "jensenshannon":
        feature_importance_distance_metric = DistanceJensenShannon(scale_features=False)
    else:
        raise ValueError(f"Distance metric {feature_importance_distance_metric} is not available!")

    feature_importance = FeatureImportance(
        distance_metric=feature_importance_distance_metric,
    )

    fi_local, fi_global, data_clustering = feature_importance.calculate_feature_importance(
        X=X, y=y, cluster_labels=cluster_labels, model_type=model_type, verbose=verbose
    )

    return Bunch(
        feature_importance_local=fi_local,
        feature_importance_global=fi_global,
        data_clustering=data_clustering,
    )


def plot_forest_guided_feature_importance(
    feature_importance_local: pd.DataFrame,
    feature_importance_global: pd.Series,
    top_n: Optional[int] = None,
    num_cols: Optional[int] = 4,
    save: Optional[str] = None,
) -> None:
    """
    Visualize global and local feature importance values as bar charts.

    The global importance represents the mean distance (importance) of each feature across
    all clusters, while the local importance shows the feature impact within each cluster.
    This function produces a series of bar plots to highlight which features drive cluster separations.

    :param feature_importance_local: Local importance values per cluster. Output of `plot_forest_guided_feature_importance()`.
    :type feature_importance_local: pandas.DataFrame
    :param feature_importance_global: Global mean importance values across clusters. Output of `plot_forest_guided_feature_importance()`.
    :type feature_importance_global: pandas.Series
    :param top_n: If specified, number of top-ranked features to plot. Default: None.
    :type top_n: Optional[int]
    :param num_cols: Number of columns in the subplot layout. Default: 4.
    :type num_cols: Optional[int]
    :param save: If specified, path prefix to save plots. Default: None.
    :type save: Optional[str]
    """
    assert isinstance(feature_importance_global, pd.Series), (
        f"Expected `feature_importance_global` to be a Series, but got {type(feature_importance_global)} "
        f"with shape {getattr(feature_importance_global, 'shape', 'N/A')}."
    )

    plot_feature_importance(
        feature_importance_local=feature_importance_local,
        feature_importance_global=feature_importance_global,
        top_n=top_n,
        num_cols=num_cols,
        save=save,
    )


def plot_forest_guided_decision_paths(
    data_clustering: pd.DataFrame,
    model_type: str,
    distributions: Optional[bool] = True,
    heatmap: Optional[bool] = True,
    heatmap_type: Optional[str] = "static",
    top_n: Optional[int] = None,
    num_cols: Optional[int] = 6,
    cmap_target_dict: Optional[dict] = None,
    save: Optional[str] = None,
) -> None:
    """
    Plot the decision patterns that emerge from forest-guided clustering using
    feature distribution plots and feature heatmaps.

    This function combines cluster labels and top-ranked features to provide
    visual insights into the decision paths formed by a Random Forest. Distribution
    plots show how feature values vary across clusters, and heatmaps summarize
    these patterns in either a static or interactive format.

    :param data_clustering: DataFrame of clustering data ordered by the global feature importance. Output of `plot_forest_guided_feature_importance()`.
    :type data_clustering: pd.DataFrame
    :param model_type: Type of model, either "cla" for classification or "reg" for regression. Output of `forest_guided_clustering()`.
    :type model_type: str
    :param distributions: Whether to show the feature distribution plots. Default: True.
    :type distributions: Optional[bool]
    :param heatmap: Whether to show the feature heatmap. Default: True.
    :type heatmap: Optional[bool]
    :param heatmap_type: Heatmap shown in "static" or "interactive" style. Default: "static".
    :type heatmap_type: Optional[str]
    :param top_n: If specified, number of top-ranked features to plot. Default: None.
    :type top_n: Optional[int]
    :param num_cols: Number of columns in the subplot layout. Default: 4.
    :type num_cols: Optional[int]
    :param cmap_target_dict: If specified, custom color map for categorical targets. Default: None.
    :type cmap_target_dict: Optional[dict]
    :param save: If specified, path prefix to save plots. Default: None.
    :type save: Optional[str]
    """
    # select top n features and cluster, target for plotting
    if top_n:
        data_clustering_selected_featues = data_clustering.iloc[:, : top_n + 2]
    else:
        data_clustering_selected_featues = data_clustering

    if distributions:
        plot_distributions(data_clustering_selected_featues, top_n, num_cols, cmap_target_dict, save)

    if heatmap:
        if model_type == "reg":
            plot_heatmap_regression(data_clustering_selected_featues, top_n, heatmap_type, save)
        elif model_type == "cla":
            plot_heatmap_classification(
                data_clustering_selected_featues, top_n, heatmap_type, cmap_target_dict, save
            )
