############################################
# Imports
############################################

import numpy as np
import pandas as pd

import seaborn as sns
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

DEFAULT_COLOR_SPEC = {
    "color_score": "#E69F00",
    "color_ji": "#0072B2",
    "color_base": "#bababa",
    "color_recolor": "tab20",
    "color_target": "Greens",
    "color_target_cat": "Greens",
    "color_features": "coolwarm",
    "color_features_cat": "Greys",
    "color_boundaries": "none",
}

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

    The fitted Random Forest is used to compute proximity-based distances between samples,
    which are then clustered with the chosen clustering strategy. The number of clusters
    can be fixed or optimized over a range of candidate values using both clustering
    stability and task-specific cluster quality. Stability is assessed with the Jaccard
    Index across repeated subsamples, while quality is measured by balanced impurity for
    classification or within-cluster variation for regression. The result is returned as a
    ``Bunch`` containing clustering diagnostics and cluster labels for each evaluated ``k``.

    :param estimator: Fitted RandomForestClassifier or RandomForestRegressor used to derive sample proximities.
    :type estimator: RandomForestClassifier | RandomForestRegressor
    :param X: Input feature matrix.
    :type X: pd.DataFrame
    :param y: Target variable, given either as target values or as the name of the target column in ``X``.
    :type y: str | pd.Series
    :param clustering_distance_metric: Distance metric based on Random Forest terminal-node proximity.
    :type clustering_distance_metric: DistanceRandomForestProximity
    :param clustering_strategy: Clustering strategy used to group samples from the distance matrix.
    :type clustering_strategy: ClusteringKMedoids | ClusteringClara
    :param k: Number of clusters if given as an integer, optimization range if given as ``(min_k, max_k)``, or ``None`` to use the default range ``(2, 6)``.
    :type k: int | tuple[int, int] | None
    :param JI_bootstrap_iter: Number of subsampling iterations used to estimate Jaccard stability.
    :type JI_bootstrap_iter: int
    :param JI_bootstrap_sample_size: Number or proportion of samples drawn in each Jaccard stability iteration. If ``None``, an adaptive subsample size is chosen.
    :type JI_bootstrap_sample_size: int | float | None
    :param JI_discart_value: Minimum mean Jaccard Index required for a clustering solution to be considered stable.
    :type JI_discart_value: float
    :param n_jobs: Number of parallel jobs used during optimization.
    :type n_jobs: int
    :param random_state: Random seed used for reproducibility.
    :type random_state: int
    :param verbose: Verbosity level controlling progress bars and printed output.
    :type verbose: int

    :return: Bunch containing the selected ``best_k``, evaluated ``ks``, mean Jaccard stability, quality scores, stability mask, per-cluster Jaccard values, cluster labels per ``k``, and the estimator class as ``model_type``.
    :rtype: Bunch
    """
    # check estimator class
    model_type = check_input_estimator(estimator)
    if model_type is None:
        raise ValueError("Model must be a scikit-learn RandomForestClassifier or RandomForestRegressor")

    # format input data
    X, y, _ = check_input_data(X, y)

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
    y_pred = estimator.predict(X)
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
    y_pred: pd.Series | None = None,
    feature_importance_distance_metric: str = "wasserstein",
    verbose: int = 1,
) -> Bunch:
    """
    Compute forest-guided feature importance by measuring how strongly features separate clusters.

    For each feature and cluster, the feature distribution inside the cluster is compared
    against the background distribution across all samples. Depending on the selected
    metric, either Wasserstein distance or Jensen-Shannon distance is used. The resulting
    local feature importance values are aggregated across clusters to obtain global
    feature importance, and the clustering table is reordered accordingly for downstream
    visualization.

    :param X: Input feature matrix.
    :type X: pd.DataFrame
    :param y: Target variable, given either as target values or as the name of the target column in ``X``.
    :type y: str | pd.Series
    :param cluster_labels: Cluster labels produced by forest-guided clustering.
    :type cluster_labels: np.ndarray
    :param y_pred: Optional predicted target values aligned with ``X`` and ``y``.
    :type y_pred: pd.Series | None
    :param feature_importance_distance_metric: Distance metric used for feature importance calculation, either ``"wasserstein"`` or ``"jensenshannon"``.
    :type feature_importance_distance_metric: str
    :param verbose: Verbosity level controlling progress output.
    :type verbose: int

    :return: Bunch containing local feature importance, global feature importance, and the clustering data reordered by global feature importance.
    :rtype: Bunch
    """
    X, y, y_pred = check_input_data(X, y, y_pred)

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
        X=X, y=y, y_pred=y_pred, cluster_labels=cluster_labels, verbose=verbose
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
    color_spec: dict | None = None,
    show: bool = True,
    save: str | None = None,
) -> tuple[Figure, Axes] | None:
    """
    Plot clustering quality and stability across the evaluated numbers of clusters.

    The plot summarizes the optimizer output by showing the clustering score, mean
    Jaccard stability, and per-cluster Jaccard stability for each evaluated value of
    ``k``. Optionally, the selected ``best_k`` and a Jaccard stability threshold are
    highlighted. Plot creation is delegated to the plotting module after merging the
    provided colors with the default color specification.

    :param ks: Evaluated numbers of clusters.
    :type ks: list[int]
    :param scores: Clustering quality score for each value in ``ks``.
    :type scores: list[float]
    :param mean_ji: Mean Jaccard stability for each value in ``ks``.
    :type mean_ji: list[float]
    :param cluster_jis: Per-cluster Jaccard stability values for each ``k``.
    :type cluster_jis: dict[int, dict[int, float]]
    :param best_k: Optional value of ``k`` to highlight in the plot.
    :type best_k: int | None
    :param JI_discart_value: Optional Jaccard stability threshold to draw as a horizontal reference line.
    :type JI_discart_value: float | None
    :param color_spec: Optional dictionary overriding entries of ``DEFAULT_COLOR_SPEC``.
    :type color_spec: dict | None
    :param show: If ``True``, display the figure; otherwise return it.
    :type show: bool
    :param save: Optional output path used to save the figure.
    :type save: str | None

    :return: Matplotlib figure and primary axes when ``show`` is ``False``; otherwise ``None``.
    :rtype: tuple[Figure, Axes] | None
    """
    color_spec = {**DEFAULT_COLOR_SPEC, **(color_spec or {})}

    return plot_optimizer_results(
        ks=ks,
        scores=scores,
        mean_ji=mean_ji,
        cluster_jis=cluster_jis,
        best_k=best_k,
        JI_discart_value=JI_discart_value,
        color_spec=color_spec,
        show=show,
        save=save,
    )


def plot_forest_guided_feature_importance(
    feature_importance_local: pd.DataFrame,
    feature_importance_global: pd.Series,
    top_n: int | None = None,
    num_cols: int = 4,
    color_spec: dict | None = None,
    reorder: bool = False,
    recolor: bool = False,
    show: bool = True,
    save: str | None = None,
) -> tuple[Figure, list[Axes]] | None:
    """
    Plot global and cluster-specific feature importance values as horizontal bar charts.

    The global feature importance summarizes the average separation strength of each
    feature across clusters, while the local feature importance values show the
    cluster-specific contribution of each feature. The resulting figure contains one
    global panel and one local panel per cluster, with optional feature reordering and
    recoloring based on global rank.

    :param feature_importance_local: Local feature importance values with features as rows and clusters as columns.
    :type feature_importance_local: pd.DataFrame
    :param feature_importance_global: Global feature importance values indexed by feature name.
    :type feature_importance_global: pd.Series
    :param top_n: Number of top-ranked features to display, or ``None`` to show all features.
    :type top_n: int | None
    :param num_cols: Maximum number of subplot columns in the figure layout.
    :type num_cols: int
    :param color_spec: Optional dictionary overriding entries of ``DEFAULT_COLOR_SPEC``.
    :type color_spec: dict | None
    :param reorder: If ``True``, order local feature panels according to the global feature ranking.
    :type reorder: bool
    :param recolor: If ``True``, color local bars according to the global feature ranking.
    :type recolor: bool
    :param show: If ``True``, display the figure; otherwise return it.
    :type show: bool
    :param save: Optional output path used to save the figure.
    :type save: str | None

    :return: Matplotlib figure and axes when ``show`` is ``False``; otherwise ``None``.
    :rtype: tuple[Figure, list[Axes]] | None
    """
    color_spec = {**DEFAULT_COLOR_SPEC, **(color_spec or {})}

    assert isinstance(feature_importance_global, pd.Series), (
        f"Expected `feature_importance_global` to be a Series, but got {type(feature_importance_global)} "
        f"with shape {getattr(feature_importance_global, 'shape', 'N/A')}."
    )

    return plot_feature_importance(
        feature_importance_local=feature_importance_local,
        feature_importance_global=feature_importance_global,
        top_n=top_n,
        num_cols=num_cols,
        color_spec=color_spec,
        reorder=reorder,
        recolor=recolor,
        show=show,
        save=save,
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
    color_spec: dict | None = None,
    show: bool = True,
    save: str | None = None,
) -> (
    tuple[
        tuple[Figure, list[Axes]] | bool | None,
        tuple[Figure, list[Axes]] | go.Figure | bool | None,
        tuple[Figure, Any] | bool | None,
    ]
    | None
):
    """
    Visualize cluster-specific decision patterns using distributions, heatmaps, and dot plots.

    This function selects the top-ranked features from ``data_clustering`` according to
    global feature importance and then produces up to three complementary visualizations:
    distribution plots, a heatmap, and a dot plot. The heatmap style can be static or
    interactive, and the regression or classification heatmap variant is selected based
    on ``model_type``. Each plot is optional and uses the merged default and user-provided
    color specification.

    :param data_clustering: Clustering table containing ``cluster``, ``target``, optional ``predicted_target``, and feature columns.
    :type data_clustering: pd.DataFrame
    :param feature_importance_global: Global feature importance values used to rank and select features for plotting.
    :type feature_importance_global: pd.Series
    :param feature_importance_local: Local feature importance values used for the dot plot.
    :type feature_importance_local: pd.DataFrame
    :param model_type: Estimator class used to choose the regression or classification heatmap variant.
    :type model_type: type[RandomForestClassifier] | type[RandomForestRegressor]
    :param distributions: If ``True``, generate cluster-wise feature distribution plots.
    :type distributions: bool
    :param heatmap: If ``True``, generate a cluster-wise heatmap.
    :type heatmap: bool
    :param heatmap_type: Heatmap rendering mode, either ``"static"`` or ``"interactive"``.
    :type heatmap_type: str
    :param dotplot: If ``True``, generate a dot plot summarizing local importance and direction of effect.
    :type dotplot: bool
    :param top_n: Number of top-ranked features to include, or ``None`` to include all ranked features.
    :type top_n: int | None
    :param num_cols: Number of subplot columns used for the distribution plot layout.
    :type num_cols: int
    :param color_spec: Optional dictionary overriding entries of ``DEFAULT_COLOR_SPEC``.
    :type color_spec: dict | None
    :param show: If ``True``, display the requested figures; otherwise return them.
    :type show: bool
    :param save: Optional output path used to save the generated figures.
    :type save: str | None

    :return: Tuple containing the requested plot objects when ``show`` is ``False``; omitted plots are returned as ``None``.
    :rtype: tuple[tuple[Figure, list[Axes]] | bool | None, tuple[Figure, list[Axes]] | go.Figure | bool | None, tuple[Figure, Any] | bool | None] | None
    """
    color_spec = {**DEFAULT_COLOR_SPEC, **(color_spec or {})}

    # select top n features and cluster, target for plotting
    feature_importance_global = feature_importance_global.sort_values(ascending=False)

    if top_n:
        feature_importance_global_selected = feature_importance_global.iloc[:top_n,]
    else:
        feature_importance_global_selected = feature_importance_global

    columns_fixed = ["cluster", "target"]
    if "predicted_target" in data_clustering.columns:
        columns_fixed.append("predicted_target")
    columns_to_select = columns_fixed + feature_importance_global_selected.index.tolist()
    data_clustering_selected_featues = data_clustering.loc[:, columns_to_select]

    if distributions:
        distributions = plot_distributions(
            data_clustering_ranked=data_clustering_selected_featues,
            top_n=top_n,
            num_cols=num_cols,
            color_spec=color_spec,
            show=show,
            save=save,
        )

    if heatmap:
        if issubclass(model_type, RandomForestRegressor):
            heatmap = plot_heatmap_regression(
                data_clustering_ranked=data_clustering_selected_featues,
                top_n=top_n,
                heatmap_type=heatmap_type,
                color_spec=color_spec,
                show=show,
                save=save,
            )
        elif issubclass(model_type, RandomForestClassifier):
            heatmap = plot_heatmap_classification(
                data_clustering_ranked=data_clustering_selected_featues,
                top_n=top_n,
                heatmap_type=heatmap_type,
                color_spec=color_spec,
                show=show,
                save=save,
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
            color_spec=color_spec,
            show=show,
            save=save,
        )

    if not show:
        return distributions, heatmap, dotplot
