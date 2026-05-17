############################################
# Imports
############################################

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from collections.abc import Sequence
from typing import Any

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from sklearn.utils import Bunch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


from .utils import check_input_data, check_input_estimator, check_sub_sample_size, check_k_range
from .clustering import ClusteringKMedoids, ClusteringClara
from .distance import DistanceWasserstein, DistanceJensenShannon, DistanceRandomForestBase
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
    clustering_distance_metric: DistanceRandomForestBase,
    clustering_strategy: ClusteringKMedoids | ClusteringClara,
    k: int | tuple[int, int] | None = None,
    JI_bootstrap_iter: int = 100,
    JI_bootstrap_sample_size: int | float | None = None,
    JI_discart_value: float = 0.6,
    n_jobs: int = 1,
    random_state: int | None = None,
    verbose: int = 1,
) -> Bunch:
    """
    Run forest-guided clustering with Random-Forest-derived distances.

    The fitted Random Forest is first encoded by ``clustering_distance_metric``. The selected
    clustering strategy then groups samples using pairwise distances derived from this
    encoding. Supported distance metrics include terminal-node proximity and LCA-based
    decision-path distances.

    The number of clusters can be fixed or optimized over a range. When a range is provided,
    candidate values are evaluated by clustering stability and task-specific cluster quality.
    Stability is estimated with bootstrapped Jaccard indices. Cluster quality is measured by
    balanced impurity for classification models and within-cluster variation for regression
    models.

    :param estimator: Fitted Random Forest estimator used to derive the sample encoding.
    :type estimator: RandomForestClassifier | RandomForestRegressor
    :param X: Input feature matrix.
    :type X: pd.DataFrame
    :param y: Target variable, either as a target vector or as the name of a column in ``X``.
    :type y: str | pd.Series
    :param clustering_distance_metric: Forest-derived distance metric used to encode samples
        and compute pairwise distances.
    :type clustering_distance_metric: DistanceRandomForestBase
    :param clustering_strategy: Clustering algorithm used with the computed distances.
    :type clustering_strategy: ClusteringKMedoids | ClusteringClara
    :param k: Fixed number of clusters, inclusive optimization range ``(min_k, max_k)``, or
        ``None`` to use the default range.
    :type k: int | tuple[int, int] | None
    :param JI_bootstrap_iter: Number of bootstrap iterations for Jaccard stability
        estimation.
    :type JI_bootstrap_iter: int
    :param JI_bootstrap_sample_size: Number or fraction of samples drawn in each Jaccard
        bootstrap iteration. If ``None``, an adaptive size is selected.
    :type JI_bootstrap_sample_size: int | float | None
    :param JI_discart_value: Minimum mean Jaccard index required for a solution to be marked
        as stable.
    :type JI_discart_value: float
    :param n_jobs: Number of parallel jobs used during cluster-number optimization.
    :type n_jobs: int
    :param random_state: Random seed used for reproducible clustering and subsampling.
    :type random_state: int | None
    :param verbose: Verbosity level for progress output.
    :type verbose: int

    :return: Results containing ``best_k``, evaluated ``ks``, mean Jaccard indices,
        quality scores, stability mask, per-cluster Jaccard values, cluster labels for each
        evaluated ``k``, and ``model_type``.
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
    clustering_distance_metric.calculate_forest_encoding(estimator=estimator, X=X)

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
    y_pred: np.ndarray | pd.Series | None = None,
    feature_importance_distance_metric: str = "wasserstein",
    verbose: int = 1,
) -> Bunch:
    """
    Compute cluster-wise and global forest-guided feature importance.

    For each feature and cluster, the feature distribution inside the cluster is compared
    with the background distribution across all samples. Local feature importance contains
    the resulting cluster-specific distances. Global feature importance is computed by
    aggregating local values across clusters.

    Supported distance metrics are ``"wasserstein"`` and ``"jensenshannon"``.

    :param X: Input feature matrix.
    :type X: pd.DataFrame
    :param y: Target variable, either as a target vector or as the name of a column in ``X``.
    :type y: str | pd.Series
    :param cluster_labels: Cluster labels aligned with ``X``.
    :type cluster_labels: np.ndarray
    :param y_pred: Optional predicted target values aligned with ``X``.
    :type y_pred: np.ndarray | pd.Series | None
    :param feature_importance_distance_metric: Distance metric used to compare cluster and
        background feature distributions. Must be ``"wasserstein"`` or
        ``"jensenshannon"``.
    :type feature_importance_distance_metric: str
    :param verbose: Verbosity level for progress output.
    :type verbose: int

    :raises ValueError: If ``feature_importance_distance_metric`` is not supported.

    :return: Results containing local feature importance, global feature importance, and the
        clustering table used for downstream visualization.
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
    ks: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    mean_ji: Sequence[float] | np.ndarray,
    cluster_jis: dict[int, dict[int, float]],
    best_k: int | None = None,
    JI_discart_value: float | None = None,
    color_spec: dict[str, Any] | None = None,
    show: bool = True,
    save: str | None = None,
) -> tuple[Figure, Axes] | None:
    """
    Plot clustering quality and stability across evaluated cluster numbers.

    The plot shows the task-specific clustering score, mean Jaccard stability, and
    per-cluster Jaccard stability for each evaluated ``k``. Optionally, the selected
    ``best_k`` and the Jaccard stability threshold are highlighted.

    :param ks: Evaluated numbers of clusters.
    :type ks: Sequence[int] | np.ndarray
    :param scores: Clustering quality score for each evaluated ``k``.
    :type scores: Sequence[float] | np.ndarray
    :param mean_ji: Mean Jaccard stability for each evaluated ``k``.
    :type mean_ji: Sequence[float] | np.ndarray
    :param cluster_jis: Per-cluster Jaccard stability values keyed by ``k``.
    :type cluster_jis: dict[int, dict[int, float]]
    :param best_k: Optional selected number of clusters to highlight.
    :type best_k: int | None
    :param JI_discart_value: Optional Jaccard stability threshold to draw.
    :type JI_discart_value: float | None
    :param color_spec: Optional overrides for ``DEFAULT_COLOR_SPEC``.
    :type color_spec: dict[str, Any] | None
    :param show: If ``True``, display the figure. If ``False``, return it.
    :type show: bool
    :param save: Optional file path for saving the figure.
    :type save: str | None

    :return: Figure and primary axes when ``show=False``; otherwise ``None``.
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
    Plot global and cluster-specific feature importance.

    The global panel summarizes feature importance across clusters. The local panels show
    cluster-specific feature importance values. Features can be limited to the top-ranked
    entries and optionally reordered or recolored according to the global ranking.

    :param feature_importance_local: Local feature importance values with features as rows
        and clusters as columns.
    :type feature_importance_local: pd.DataFrame
    :param feature_importance_global: Global feature importance values indexed by feature
        name.
    :type feature_importance_global: pd.Series
    :param top_n: Number of top-ranked features to display, or ``None`` to display all.
    :type top_n: int | None
    :param num_cols: Maximum number of subplot columns.
    :type num_cols: int
    :param color_spec: Optional overrides for ``DEFAULT_COLOR_SPEC``.
    :type color_spec: dict | None
    :param reorder: If ``True``, order local feature panels by global feature ranking.
    :type reorder: bool
    :param recolor: If ``True``, color local bars by global feature ranking.
    :type recolor: bool
    :param show: If ``True``, display the figure. If ``False``, return it.
    :type show: bool
    :param save: Optional file path for saving the figure.
    :type save: str | None

    :return: Figure and axes when ``show=False``; otherwise ``None``.
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
    draw_distributions: bool = True,
    draw_dotplot: bool = True,
    draw_heatmap: bool = True,
    heatmap_type: str = "static",
    top_n: int | None = None,
    num_cols: int = 6,
    color_spec: dict | None = None,
    show: bool = True,
    save: str | None = None,
) -> (
    tuple[
        tuple[Figure, list[Axes]] | None,
        tuple[Figure, list[Axes]] | None,
        tuple[Figure, list[Axes]] | go.Figure | None,
    ]
    | None
):
    """
    Plot cluster-specific decision patterns for important features.

    Features are ranked by global feature importance, optionally restricted to ``top_n``,
    and visualized with up to three complementary plots: feature distributions, a dot plot,
    and a heatmap. The heatmap variant is selected from ``model_type`` to support both
    regression and classification outputs.

    :param data_clustering: Clustering table containing ``cluster``, ``target``, optional
        ``predicted_target``, and feature columns.
    :type data_clustering: pd.DataFrame
    :param feature_importance_global: Global feature importance values used to rank features.
    :type feature_importance_global: pd.Series
    :param feature_importance_local: Local feature importance values used for the dot plot.
    :type feature_importance_local: pd.DataFrame
    :param model_type: Random Forest estimator class used to select the heatmap variant.
    :type model_type: type[RandomForestClassifier] | type[RandomForestRegressor]
    :param draw_distributions: If ``True``, generate feature distribution plots.
    :type draw_distributions: bool
    :param draw_dotplot: If ``True``, generate the local-importance dot plot.
    :type draw_dotplot: bool
    :param draw_heatmap: If ``True``, generate the cluster-wise heatmap.
    :type draw_heatmap: bool
    :param heatmap_type: Heatmap rendering mode, for example ``"static"`` or
        ``"interactive"``.
    :type heatmap_type: str
    :param top_n: Number of top-ranked features to plot, or ``None`` to plot all.
    :type top_n: int | None
    :param num_cols: Maximum number of columns for distribution subplots.
    :type num_cols: int
    :param color_spec: Optional overrides for ``DEFAULT_COLOR_SPEC``.
    :type color_spec: dict | None
    :param show: If ``True``, display the generated figures. If ``False``, return them.
    :type show: bool
    :param save: Optional file path or prefix for saving generated figures.
    :type save: str | None

    :raises ValueError: If ``model_type`` is not a Random Forest classifier or regressor.

    :return: Distribution, dot-plot, and heatmap outputs when ``show=False``. Disabled plots
        are returned as ``None``. Returns ``None`` when ``show=True``.
    :rtype: tuple[tuple[Figure, list[Axes]] | None, tuple[Figure, list[Axes]] | None, tuple[Figure, list[Axes]] | go.Figure | None] | None
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
    data_clustering_selected_features = data_clustering.loc[:, columns_to_select]

    if draw_distributions:
        distributions_out = plot_distributions(
            data_clustering_ranked=data_clustering_selected_features,
            top_n=top_n,
            num_cols=num_cols,
            color_spec=color_spec,
            show=show,
            save=save,
        )
    else:
        distributions_out = None

    if draw_dotplot:
        dotplot_out = plot_dotplot(
            data_clustering_ranked=data_clustering_selected_features,
            feature_importance_global=feature_importance_global_selected,
            feature_importance_local=feature_importance_local,
            top_n=top_n,
            color_spec=color_spec,
            show=show,
            save=save,
        )
    else:
        dotplot_out = None

    if draw_heatmap:
        if issubclass(model_type, RandomForestRegressor):
            heatmap_out = plot_heatmap_regression(
                data_clustering_ranked=data_clustering_selected_features,
                top_n=top_n,
                heatmap_type=heatmap_type,
                color_spec=color_spec,
                show=show,
                save=save,
            )
        elif issubclass(model_type, RandomForestClassifier):
            heatmap_out = plot_heatmap_classification(
                data_clustering_ranked=data_clustering_selected_features,
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
    else:
        heatmap_out = None

    if not show:
        return distributions_out, dotplot_out, heatmap_out
