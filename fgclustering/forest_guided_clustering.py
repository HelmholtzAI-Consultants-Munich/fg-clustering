############################################
# Imports
############################################

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Any

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from sklearn.utils import Bunch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


from .utils import check_input_data, check_input_estimator, check_sub_sample_size, check_k_range
from .clustering import ClusteringKMedoids, ClusteringClara
from .distance import DistanceWasserstein, DistanceJensenShannon, DistanceRandomForestProximity
from .optimizer import Optimizer
from .statistics import FeatureImportance
from .plotting import (
    plot_optimizer_results,
    plot_feature_importance,
    plot_distributions,
    plot_heatmap_regression,
    plot_heatmap_classification,
    plot_dotplot,
)


############################################
# Forest-guided Clustering API
############################################


def forest_guided_clustering(
    estimator: RandomForestClassifier | RandomForestRegressor,
    X: pd.DataFrame,
    y: str | pd.Series,
    clustering_distance_metric: DistanceRandomForestProximity,
    clustering_strategy: ClusteringKMedoids | ClusteringClara,
    k: int | tuple[int, int] | None = None,
    JI_bootstrap_iter: int = 100,
    JI_bootstrap_sample_size: int | float | None = None,
    JI_discart_value: float = 0.6,
    n_jobs: int = 1,
    random_state: int = 42,
    verbose: int = 1,
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
    :type estimator: RandomForestClassifier | RandomForestRegressor
    :param X: Input feature matrix.
    :type X: pd.DataFrame
    :param y: Target variable, i.e. target values or name of the target column in X.
    :type y: str | pd.Series
    :param clustering_distance_metric: An instance of DistanceRandomForestProximity.
    :type clustering_distance_metric: DistanceRandomForestProximity
    :param clustering_strategy: An instance of ClusteringKMedoids or ClusteringClara.
    :type clustering_strategy: ClusteringKMedoids | ClusteringClara
    :param k: If int, number of clusters is fixed. If tuple, number of clusters is optimized within the given range. If None number of clusters is optimized within the range of 2 to 6. Default: None
    :type k: int | tuple[int, int] | None
    :param JI_bootstrap_iter: Number of bootstrap iterations for Jaccard Index evaluation. Default: 100
    :type JI_bootstrap_iter: int
    :param JI_bootstrap_sample_size: Number or proportion of samples to draw for each JI bootstrap. If None, computes an adaptive subsample ratio based on dataset size, constrained between 10% and 80%, targeting approximately 1,000 samples. Default: None.
    :type JI_bootstrap_sample_size: int | float | None
    :param JI_discart_value: Jaccard Index threshold to discard unstable clusters. Default: 0.6
    :type JI_discart_value: float
    :param n_jobs: Number of parallel jobs. Default: 1.
    :type n_jobs: int
    :param random_state: Random seed for reproducibility. Default: 42.
    :type random_state: int
    :param verbose: Verbosity level (0 = silent, 1 = progress messages). Default: 1.
    :type verbose: int

    :return: A Bunch with best_k, ks, mean_ji, scores, stable_mask, cluster_jis, cluster_labels (dict by k), and model_type (estimator class, e.g. ``RandomForestClassifier``).
    :rtype: Bunch
    """
    # check estimator class
    model_type = check_input_estimator(estimator)
    if model_type is None:
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
    results, best_k = optimizer.optimizeK(
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
        best_k=best_k,
        ks=np.array([r["k"] for r in results]),
        mean_ji=np.array([r["Mean_JI"] for r in results]),
        scores=np.array([r["Score"] for r in results]),
        stable_mask=np.array([r["Stable"] for r in results]),
        cluster_jis={r["k"]: r["Cluster_JI"] for r in results},
        cluster_labels={r["k"]: r["Cluster_labels"] for r in results},
        model_type=model_type,
    )


def forest_guided_feature_importance(
    X: pd.DataFrame,
    y: str | pd.Series,
    cluster_labels: np.ndarray,
    model_type: type[RandomForestClassifier] | type[RandomForestRegressor],
    feature_importance_distance_metric: str = "wasserstein",
    verbose: int = 1,
) -> Bunch:
    """
    Compute forest-guided feature importance by quantifying how much each feature contributes to cluster separation.

    This function compares the distribution of each feature within a cluster to its distribution across the entire dataset,
    using either the Wasserstein or Jensen-Shannon distance metric. Wasserstein distance is recommended for a dataset with primarily
    continuous features, while Jensen-Shannon is better suited for categorical features. The resulting scores indicate
    which features are most important for distinguishing clusters, providing interpretability into the model’s decision structure.

    :param X: Input feature matrix.
    :type X: pd.DataFrame
    :param y: Target variable, i.e. target values or name of the target column in X.
    :type y: str | pd.Series
    :param cluster_labels: Labels from forest-guided clustering. Output of `forest_guided_clustering()`.
    :type cluster_labels: np.ndarray
    :param model_type: Estimator class from ``forest_guided_clustering`` (``RandomForestClassifier`` or ``RandomForestRegressor``, or a subclass). Compare with e.g. ``model_type is RandomForestClassifier``.
    :type model_type: type[RandomForestClassifier] | type[RandomForestRegressor]
    :param feature_importance_distance_metric: Distance metric for computing feature importance ("wasserstein" or "jensenshannon"). Default: "wasserstein".
    :type feature_importance_distance_metric: str
    :param verbose: Verbosity level (0 = silent, 1 = progress messages). Default: 1.
    :type verbose: int

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


def plot_forest_guided_clustering(
    ks: list[int],
    scores: list[float],
    mean_ji: list[float],
    cluster_jis: dict[int, dict[int, float]],
    best_k: int | None = None,
    JI_discart_value: float | None = None,
    show: bool = True,
    save: str | None = None,
    cmap: dict[str, Any] | None = None,
) -> tuple[Figure, Axes] | None:
    """
    Plot optimizer results: clustering score, mean Jaccard stability, and per-cluster stability.

    Produces a figure with number of clusters (k) on the x-axis, mean cluster stability and
    per-cluster stability on the left y-axis, and clustering score on the right y-axis.
    Optionally marks the best k and a stability threshold. Delegates to the plotting module.

    :param ks: Number of clusters evaluated (one value per run).
    :type ks: list[int]
    :param scores: Clustering score for each k (same length as ks).
    :type scores: list[float]
    :param mean_ji: Mean Jaccard stability (cluster stability) for each k (same length as ks).
    :type mean_ji: list[float]
    :param cluster_jis: Per-cluster Jaccard stability for each k. Keys are values from ks; each value is a dict mapping cluster id to stability (float).
    :type cluster_jis: dict[int, dict[int, float]]
    :param best_k: If set, a vertical line and label are drawn at this k. Default is None.
    :type best_k: int | None
    :param JI_discart_value: If set, a horizontal line is drawn at this stability threshold. Default is None.
    :type JI_discart_value: float | None
    :param show: If True, call ``matplotlib.pyplot.show()`` before returning. If False, return the figure and axes so the plot can be customized further. Default is True.
    :type show: bool
    :param save: If set, the figure is saved using this path base (with an "_optimizer_results" suffix).
    :type save: str | None
    :param cmap: Color map; cmap.values() are passed to seaborn's color palette. If None, the "colorblind" palette is used. Default is None.
    :type cmap: dict[str, Any] | None

    :return: The Matplotlib figure and main axes (left y-axis) if ``show`` is False; otherwise None.
    :rtype: tuple[Figure, Axes] | None
    """
    return plot_optimizer_results(
        ks=ks,
        scores=scores,
        mean_ji=mean_ji,
        cluster_jis=cluster_jis,
        best_k=best_k,
        JI_discart_value=JI_discart_value,
        show=show,
        save=save,
        cmap=cmap,
    )


def plot_forest_guided_feature_importance(
    feature_importance_local: pd.DataFrame,
    feature_importance_global: pd.Series,
    top_n: int | None = None,
    num_cols: int = 4,
    save: str | None = None,
    show: bool = True,
    reorder: bool = False,
    recolor: bool = False,
) -> tuple[Figure, list[Axes]] | None:
    """
    Visualize global and local feature importance values as bar charts.

    The global importance represents the mean distance (importance) of each feature across
    all clusters, while the local importance shows the feature impact within each cluster.
    This function produces a series of bar plots to highlight which features drive cluster separations.

    :param feature_importance_local: Local importance values per cluster. Output of `plot_forest_guided_feature_importance()`.
    :type feature_importance_local: pd.DataFrame
    :param feature_importance_global: Global mean importance values across clusters. Output of `plot_forest_guided_feature_importance()`.
    :type feature_importance_global: pd.Series
    :param top_n: If specified, number of top-ranked features to plot. Default: None.
    :type top_n: int | None
    :param num_cols: Number of columns in the subplot layout. Default: 4.
    :type num_cols: int
    :param save: If set, the figure is saved using this path base (with a "_feature_importance" suffix).
    :type save: str | None
    :param show: If True, call ``matplotlib.pyplot.show()`` before returning. If False, return the figure and axes so the plot can be customized further. Default is True.
    :type show: bool
    :param reorder: If True, reorder the local importance values to match the global importance order.
    :type reorder: bool
    :param recolor: If True, recolor the bars based on the global importance order.
    :type recolor: bool

    :return: Matplotlib figure and axes with bar charts of global and local feature importance values if ``show`` is False; otherwise None.
    :rtype: tuple[Figure, list[Axes]] | None
    """
    assert isinstance(feature_importance_global, pd.Series), (
        f"Expected `feature_importance_global` to be a Series, but got {type(feature_importance_global)} "
        f"with shape {getattr(feature_importance_global, 'shape', 'N/A')}."
    )

    return plot_feature_importance(
        feature_importance_local=feature_importance_local,
        feature_importance_global=feature_importance_global,
        top_n=top_n,
        num_cols=num_cols,
        save=save,
        show=show,
        reorder=reorder,
        recolor=recolor,
    )


def plot_forest_guided_decision_paths(
    data_clustering: pd.DataFrame,
    feature_importance_global: pd.Series,
    feature_importance_local: pd.DataFrame,
    model_type: type[RandomForestClassifier] | type[RandomForestRegressor],
    distributions: bool = True,
    heatmap: bool = True,
    heatmap_type: str = "static",
    dotplot: bool = True,
    top_n: int | None = None,
    num_cols: int = 6,
    cmap_target_dict: dict | None = None,
    save: str | None = None,
    show: bool = True,
) -> tuple[tuple[Figure, list[Axes]] | None, (tuple[Figure, list[Axes]] | go.Figure) | None]:
    """
    Plot the decision patterns that emerge from forest-guided clustering using
    feature distribution plots and feature heatmaps.

    This function combines cluster labels and top-ranked features to provide
    visual insights into the decision paths formed by a Random Forest. Distribution
    plots show how feature values vary across clusters, and heatmaps summarize
    these patterns in either a static or interactive format.

    :param data_clustering: DataFrame of clustering data ordered by the global feature importance. Output of `plot_forest_guided_feature_importance()`.
    :type data_clustering: pd.DataFrame
    :param model_type: Estimator class from ``forest_guided_clustering`` (``RandomForestClassifier`` or ``RandomForestRegressor``, or a subclass).
    :type model_type: type[RandomForestClassifier] | type[RandomForestRegressor]
    :param distributions: Whether to show the feature distribution plots. Default: True.
    :type distributions: bool
    :param heatmap: Whether to show the feature heatmap. Default: True.
    :type heatmap: bool
    :param heatmap_type: Heatmap shown in "static" or "interactive" style. Default: "static".
    :type heatmap_type: str
    :param dotplot: Whether to show the dotplot. Default: True.
    :type dotplot: bool
    :param top_n: If specified, number of top-ranked features to plot. Default: None.
    :type top_n: int | None
    :param num_cols: Number of columns in the subplot layout. Default: 6.
    :type num_cols: int
    :param cmap_target_dict: If specified, custom color map for categorical targets. Default: None.
    :type cmap_target_dict: dict | None
    :param save: If set, figures are saved using this path base (with a "_boxplots" suffix for distribution plots; with a "_heatmap" suffix for static heatmaps, or as "{stem}_interactive_heatmap.html" in the same directory for interactive heatmaps).
    :type save: str | None
    :param show: If True, display figures before returning (matplotlib: ``plt.show()``; interactive Plotly: ``fig.show()``). If False, return figure objects for further customization. Default is True.
    :type show: bool

    :return: Tuple of two optional plots (each None if the corresponding plot was not requested or if ``show`` is True). The first can be a matplotlib `fig, axes` pair; the second can be matplotlib axes or an interactive plotly Figure.
    :rtype: tuple[tuple[Figure, list[Axes]] | None, (tuple[Figure, list[Axes]] | go.Figure) | None]
    """
    # select top n features and cluster, target for plotting
    if top_n:
        data_clustering_selected_featues = data_clustering.iloc[:, : top_n + 2]
        feature_importance_global_selected = feature_importance_global.iloc[:top_n,]
    else:
        data_clustering_selected_featues = data_clustering
        feature_importance_global_selected = feature_importance_global

    if distributions:
        distributions = plot_distributions(
            data_clustering_ranked=data_clustering_selected_featues,
            top_n=top_n,
            num_cols=num_cols,
            cmap_target_dict=cmap_target_dict,
            save=save,
            show=show,
        )

    if heatmap:
        if issubclass(model_type, RandomForestRegressor):
            heatmap = plot_heatmap_regression(
                data_clustering_ranked=data_clustering_selected_featues,
                top_n=top_n,
                heatmap_type=heatmap_type,
                save=save,
                show=show,
            )
        elif issubclass(model_type, RandomForestClassifier):
            heatmap = plot_heatmap_classification(
                data_clustering_ranked=data_clustering_selected_featues,
                top_n=top_n,
                heatmap_type=heatmap_type,
                cmap_target_dict=cmap_target_dict,
                save=save,
                show=show,
            )
        else:
            raise ValueError(
                "model_type must be RandomForestClassifier or RandomForestRegressor (or a subclass thereof)."
            )

    if dotplot:
        dotplot = plot_dotplot(
            data_clustering_ranked=data_clustering_selected_featues,
            feature_importance_global=feature_importance_global_selected,
            feature_importance_local=feature_importance_local,
            top_n=top_n,
            save=save,
            show=show,
        )

    return distributions, heatmap, dotplot
