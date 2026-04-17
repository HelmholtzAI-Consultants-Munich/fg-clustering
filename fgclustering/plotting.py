############################################
# Imports
############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, to_rgba
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Any

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

from .utils import matplotlib_to_plotly, save_figure

CLIP_ZSCORE_NEGATIVE = -3
CLIP_ZSCORE_POSITIVE = 3

############################################
# Plotting Functions
############################################


def plot_optimizer_results(
    ks: list[int],
    scores: list[float],
    mean_ji: list[float],
    cluster_jis: dict[int, dict[int, float]],
    best_k: int | None,
    JI_discart_value: float | None,
    color_spec: dict,
    show: bool,
    save: str | None,
) -> tuple[Figure, Axes] | None:
    """
    Summarize clustering quality across candidate values of ``k``.

    For each ``k``, plots the clustering score (triangles), mean Jaccard stability
    (line with markers), a translucent min–max stability band, and faint per-cluster
    Jaccard values. A secondary y-axis labels the clustering score separately from
    stability. Optionally marks ``best_k`` with a vertical line and
    ``JI_discart_value`` with a horizontal cutoff.

    :param ks: Candidate numbers of clusters evaluated by the optimizer, in plotting order.
    :type ks: list[int]
    :param scores: Clustering score for each value in ``ks``; must be aligned with ``ks``.
    :type scores: list[float]
    :param mean_ji: Mean Jaccard stability for each value in ``ks``; must be aligned with ``ks``.
    :type mean_ji: list[float]
    :param cluster_jis: Mapping from each ``k`` to per-cluster Jaccard stability values.
    :type cluster_jis: dict[int, dict[int, float]]
    :param best_k: Value of ``k`` to highlight with a vertical reference line when present in ``ks``.
    :type best_k: int | None
    :param JI_discart_value: Optional horizontal reference line for the Jaccard stability threshold.
    :type JI_discart_value: float | None
    :param color_spec: Color specification dictionary containing ``"color_score"`` and ``"color_ji"``.
    :type color_spec: dict
    :param show: If ``True``, display the figure with ``plt.show()``; otherwise return it.
    :type show: bool
    :param save: Base path with extension; saves ``{stem}_optimizer_results{suffix}`` when provided.
    :type save: str | None

    :return: ``(fig, ax)`` when ``show`` is ``False``; otherwise ``None``.
    :rtype: tuple[Figure, Axes] | None
    """

    ### Data Processing
    summary_rows = []
    cluster_rows = []

    for k, score, mean_ji in zip(ks, scores, mean_ji):
        ji_vals = list(cluster_jis[k].values())

        summary_rows.append(
            {
                "k": k,
                "score": score,
                "mean_ji": mean_ji,
                "min_ji": min(ji_vals),
                "max_ji": max(ji_vals),
            }
        )

        for ji in ji_vals:
            cluster_rows.append({"k": k, "ji": ji})

    df_summary = pd.DataFrame(summary_rows)
    df_clusters = pd.DataFrame(cluster_rows)

    ### Plotting
    sns.set_theme(style="white", context="paper")

    fig_width = max(6.5, len(ks) / 2)
    fig_height = max(4.5, len(ks) / 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plt.suptitle("Clustering Performance Across Number of Clusters (k)", fontsize=14)

    sns.scatterplot(
        data=df_summary,
        x="k",
        y="score",
        ax=ax,
        marker="^",
        s=70,
        color=color_spec["color_score"],
        zorder=3,
    )
    sns.lineplot(
        data=df_summary,
        x="k",
        y="mean_ji",
        ax=ax,
        color=color_spec["color_ji"],
        linewidth=1.5,
        marker="o",
        markersize=7,
        zorder=4,
    )
    sns.scatterplot(
        data=df_clusters,
        x="k",
        y="ji",
        ax=ax,
        s=20,
        alpha=0.25,
        color=color_spec["color_ji"],
        zorder=2,
    )

    ax.fill_between(
        df_summary["k"],
        df_summary["min_ji"],
        df_summary["max_ji"],
        color=color_spec["color_ji"],
        alpha=0.15,
        linewidth=0,
        zorder=1,
    )

    ax.set_xlabel("Number of clusters $k$")
    ax.set_ylabel("Cluster Stability", color=color_spec["color_ji"])
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="y", colors=color_spec["color_ji"])
    ax.spines["left"].set_color(color_spec["color_ji"])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax_r = ax.twinx()
    ax_r.set_ylabel("Clustering Score", color=color_spec["color_score"])
    ax_r.set_ylim(0, 1.02)
    ax_r.tick_params(axis="y", colors=color_spec["color_score"])
    ax_r.spines["right"].set_color(color_spec["color_score"])
    ax_r.spines["left"].set_visible(False)
    ax_r.spines["top"].set_visible(False)

    if best_k is not None and best_k in df_summary["k"].values:
        ax.axvline(best_k, linestyle=":", linewidth=1, color="0.5")
        ax.text(best_k, 0.5, f"k = {best_k}", rotation=90, ha="right", va="top", fontsize=9, color="0.5")

    if JI_discart_value is not None:
        ax.axhline(JI_discart_value, linestyle="--", linewidth=0.7, color=color_spec["color_ji"])

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if save:
        save_figure(save, "_optimizer_results")
    if show:
        plt.show()
    else:
        return fig, ax


def plot_feature_importance(
    feature_importance_local: pd.DataFrame,
    feature_importance_global: pd.Series,
    top_n: int | None,
    num_cols: int,
    color_spec: dict,
    reorder: bool,
    recolor: bool,
    show: bool,
    save: str | None,
) -> tuple[Figure, list[Axes]] | None:
    """
    Compare global and cluster-specific feature importance using horizontal bar plots.

    The first subplot shows global feature importance, and each additional subplot shows
    local feature importance for one cluster. When ``reorder`` is ``True``, local plots
    follow the feature order of the global ranking. When ``recolor`` is ``True``, local
    bars are colored according to the global feature rank.

    :param feature_importance_local: DataFrame with features as rows and clusters as columns.
    :type feature_importance_local: pd.DataFrame
    :param feature_importance_global: Global feature importance values indexed by feature name.
    :type feature_importance_global: pd.Series
    :param top_n: Number of top globally ranked features to display; ``None`` keeps all features.
    :type top_n: int | None
    :param num_cols: Maximum number of subplot columns in the figure grid.
    :type num_cols: int
    :param color_spec: Color specification dictionary containing at least ``"color_base"`` and,
        when ``recolor`` is enabled, ``"color_recolor"``.
    :type color_spec: dict
    :param reorder: If ``True``, local plots use the same feature ordering as the global plot.
    :type reorder: bool
    :param recolor: If ``True``, color local bars by global feature rank instead of using a single color.
    :type recolor: bool
    :param show: If ``True``, display the figure with ``plt.show()``; otherwise return it.
    :type show: bool
    :param save: Base path with extension; saves ``{stem}_feature_importance{suffix}`` when provided.
    :type save: str | None

    :return: ``(fig, fig.axes)`` when ``show`` is ``False``; otherwise ``None``.
    :rtype: tuple[Figure, list[Axes]] | None
    """

    def draw_bar(idx, data, title):
        ax = plt.subplot(num_rows, num_cols, idx)
        sns.barplot(data=data, x="Importance", y="Feature", orient="h", **kwargs)
        ax.set_xlim(0, 1)
        ax.set_title(title)

    ### Data Processing

    n_global = len(feature_importance_global.index)
    num_features = min(n_global, top_n) if top_n is not None else n_global
    num_subplots = 1 + feature_importance_local.shape[1]
    num_cols = min(num_cols, num_subplots)
    num_rows = int(np.ceil(num_subplots / num_cols))

    importance_global = pd.DataFrame(
        {
            "Feature": feature_importance_global.index,
            "Importance": feature_importance_global.to_list(),
        }
    ).sort_values(by="Importance", ascending=False)

    if top_n:
        importance_global = importance_global.iloc[:top_n,]

    if recolor:
        feats = importance_global["Feature"].tolist()
        kwargs = {
            "hue": "Feature",
            "palette": dict(zip(feats, sns.color_palette(color_spec["color_recolor"], n_colors=len(feats)))),
        }
    else:
        kwargs = {"color": color_spec["color_base"]}

    ### Plotting

    sns.set_theme(style="white", context="paper")

    fig_width = num_cols * 4.5
    fig_height = num_rows * max(4.5, int(np.ceil(5 * num_features / 25)))
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.subplots_adjust(top=0.95, hspace=0.8, wspace=0.8)
    plt.suptitle(
        f"Feature Importance: Global vs Cluster-Level {'(top ' + str(top_n) + 'features)' if top_n else ''}",
        fontsize=14,
    )

    draw_bar(1, importance_global, "Global Feature Importance")

    for n, cluster in enumerate(feature_importance_local.columns):
        if reorder:
            importance_local = (
                feature_importance_local[cluster]
                .iloc[importance_global.index]
                .rename_axis("Feature")
                .reset_index(name="Importance")
            )
        else:
            importance_local = (
                feature_importance_local[cluster]
                .sort_values(ascending=False)
                .rename_axis("Feature")
                .reset_index(name="Importance")
            )
        if top_n:
            importance_local = importance_local.iloc[:top_n]
        draw_bar(n + 2, importance_local, f"Local Feature Importance - Cluster {cluster}")

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if save:
        save_figure(save, "_feature_importance")
    if show:
        plt.show()
    else:
        return fig, fig.axes


def plot_distributions(
    data_clustering_ranked: pd.DataFrame,
    top_n: int | None,
    num_cols: int,
    color_spec: dict,
    show: bool,
    save: str | None,
) -> tuple[Figure, list[Axes]] | None:
    """
    Plot feature and target distributions by cluster in a subplot grid.

    Each column in ``data_clustering_ranked`` other than ``cluster`` is visualized in its
    own subplot. Continuous variables are shown as boxplots grouped by cluster. Discrete
    variables are shown either as count plots (for ``target`` and ``predicted_target``) or
    as stacked percentage bar plots (for other discrete features). The column order of the
    input frame defines the subplot order.

    :param data_clustering_ranked: DataFrame containing a ``cluster`` column and the variables to plot.
    :type data_clustering_ranked: pd.DataFrame
    :param top_n: Number of top-ranked features shown in the title only; does not subset the input.
    :type top_n: int | None
    :param num_cols: Number of subplot columns in the figure grid.
    :type num_cols: int
    :param color_spec: Color specification dictionary containing ``"color_base"``,
        ``"color_target_cat"``, and ``"color_features_cat"``.
    :type color_spec: dict
    :param show: If ``True``, display the figure with ``plt.show()``; otherwise return it.
    :type show: bool
    :param save: Base path with extension; saves ``{stem}_boxplots{suffix}`` when provided.
    :type save: str | None

    :return: ``(fig, fig.axes)`` when ``show`` is ``False``; otherwise ``None``.
    :rtype: tuple[Figure, list[Axes]] | None
    """

    ### Data Processing

    df = data_clustering_ranked

    features_to_plot = df.drop(columns="cluster").columns.to_list()
    num_rows = int(np.ceil(len(features_to_plot) / num_cols))

    ### Plotting
    sns.set_theme(style="white", context="paper")

    fig_width = num_cols * 4.5
    fig_height = num_rows * 4.5

    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.subplots_adjust(top=0.95, hspace=0.8, wspace=0.8)
    plt.suptitle(
        f"Distribution of Feature Values by Cluster {'(top ' + str(top_n) + 'features)' if top_n else ''}",
        fontsize=14,
    )

    for n, feature in enumerate(features_to_plot):
        ax = plt.subplot(num_rows, num_cols, n + 1)
        discrete = df[feature].nunique() < 5 or isinstance(df[feature].dtype, pd.CategoricalDtype)

        if not discrete:
            sns.boxplot(x="cluster", y=feature, data=df, ax=ax, color=color_spec["color_base"], orient="v")
            ax.set_title(f"{feature}")
            continue

        if feature in ["target", "predicted_target"]:
            sns.countplot(
                x="cluster",
                hue=feature,
                data=df,
                ax=ax,
                palette=sns.color_palette(
                    color_spec["color_target_cat"],
                    n_colors=df[feature].nunique(),
                    as_cmap=False,
                ),
            )
            ax.set_title(f"{feature}")
            ax.legend(bbox_to_anchor=(1, 1), loc=2, fontsize="x-small")
            continue

        df[feature] = df[feature].astype("string")
        count_df = df.groupby(["cluster", feature], observed=False).size().unstack(fill_value=0)
        top_categories = count_df.sum().nlargest(10).index
        count_df = count_df[
            top_categories.tolist() + [c for c in count_df.columns if c not in top_categories]
        ]
        percent_df = count_df.div(count_df.sum(axis=1), axis=0) * 100
        percent_df.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=0.8,
            color=sns.color_palette(
                color_spec["color_features_cat"],
                n_colors=df[feature].nunique(),
                as_cmap=False,
            ),
            legend=False,
        )
        ax.set_title(f"{feature}")
        ax.set_ylabel("percentage")
        ax.set_xlabel("cluster")

        handles, labels = ax.get_legend_handles_labels()
        top_indices = [i for i, label in enumerate(labels) if label in top_categories]
        ax.legend(
            [handles[i] for i in top_indices],
            [labels[i] for i in top_indices],
            bbox_to_anchor=(1, 1),
            loc=2,
            fontsize="x-small",
            title="Category",
        )

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if save:
        save_figure(save, "_boxplots")
    if show:
        plt.show()
    else:
        return fig, fig.axes


def plot_dotplot(
    data_clustering_ranked: pd.DataFrame,
    feature_importance_global: pd.Series,
    feature_importance_local: pd.DataFrame,
    top_n: int | None,
    color_spec: dict,
    show: bool,
    save: str | None,
):
    """
    Plot a dot plot of cluster-wise feature summaries across the global feature ranking.

    Each point represents one ``(feature, cluster)`` pair. The x-axis gives the global
    feature rank, the y-axis gives the cluster, the point color encodes the mean
    standardized feature value within that cluster, and the point size encodes the local
    feature importance. A color bar and a size legend are added to the figure.

    :param data_clustering_ranked: DataFrame containing ``cluster`` and feature columns;
        may also include ``target`` and ``predicted_target``, which are ignored here.
    :type data_clustering_ranked: pd.DataFrame
    :param feature_importance_global: Global feature importance values indexed by feature name;
        defines feature order on the x-axis.
    :type feature_importance_global: pd.Series
    :param feature_importance_local: Local feature importance values with features as rows and clusters as columns.
    :type feature_importance_local: pd.DataFrame
    :param top_n: Number of top-ranked features shown in the title only; does not subset the input.
    :type top_n: int | None
    :param color_spec: Color specification dictionary containing ``"color_features"`` for the dot color scale.
    :type color_spec: dict
    :param show: If ``True``, display the figure with ``plt.show()``; otherwise return it.
    :type show: bool
    :param save: Base path with extension; saves ``{stem}_dotplot{suffix}`` when provided.
    :type save: str | None

    :return: ``(fig, axes)`` when ``show`` is ``False``; otherwise ``None``.
    :rtype: tuple[Figure, np.ndarray] | None
    """
    ### Data Processing

    meta_cols = [c for c in ["cluster", "target", "predicted_target"] if c in data_clustering_ranked.columns]
    features = data_clustering_ranked.drop(columns=meta_cols)
    features = _process_features_for_plotting(features)
    features["cluster"] = data_clustering_ranked["cluster"]

    fi_local = feature_importance_local.loc[feature_importance_global.index]

    avgs = features.groupby("cluster").mean(numeric_only=True).T
    avgs = avgs.loc[feature_importance_global.index]
    avgs["global_rank"] = range(1, len(avgs) + 1)

    melted = (
        avgs.reset_index()
        .rename(columns={"index": "feature"})
        .melt(id_vars=["feature", "global_rank"], var_name="cluster", value_name="feature_avg")
        .merge(
            fi_local.reset_index()
            .rename(columns={"index": "feature"})
            .melt(id_vars="feature", var_name="cluster", value_name="local_importance"),
            on=["feature", "cluster"],
        )
    )

    n_features = avgs.shape[0]
    n_clusters = avgs.shape[1] - 1

    ### Plotting
    sns.set_theme(style="white", context="paper")

    # Match largest scatter marker (sizes=(5, 200) → 200 pt²): ~constant pt spacing per rank/cluster.
    d_max = 2 * np.sqrt(200 / np.pi)
    g, w_ax_frac, h_ax_frac = 1.22, 0.84 * (30 / 34), 0.58
    fig_width = max(5.0, g * d_max * max(n_features, 1) / (72 * w_ax_frac) + 2.5)
    fig_height = max(3.5, g * d_max * max(n_clusters, 1) / (72 * h_ax_frac))

    # Keep colorbar + size-legend columns ~fixed width (in); only the main panel scales with fig_width.
    _r_cbar, _r_leg = 2.0, 2.5
    _r_side = _r_cbar + _r_leg
    _cbar_w_in = 0.35
    _r_main = max(1.0, fig_width * _r_cbar / _cbar_w_in - _r_side)

    fig, axes = plt.subplots(
        figsize=(fig_width, fig_height),
        ncols=3,
        width_ratios=[_r_main, _r_cbar, _r_leg],
        facecolor="none",
    )
    plt.subplots_adjust(top=0.95, hspace=0.8, wspace=0.8)
    plt.suptitle(
        f"Importance and Direction of Effect by Cluster {'(top ' + str(top_n) + 'features)' if top_n else ''}",
        fontsize=14,
    )

    ax_plot, ax_cbar, ax_legend = axes

    ax_plot.set_aspect("auto")
    sns.despine(ax=ax_plot, left=True, bottom=True)

    scatter = sns.scatterplot(
        data=melted,
        x="global_rank",
        y="cluster",
        hue="feature_avg",
        size="local_importance",
        palette=sns.color_palette(color_spec["color_features"], as_cmap=True),
        sizes=(5, 200),
        size_norm=(0, 1),
        legend=False,
        ax=ax_plot,
    )

    ax_plot.set(
        xlim=(0.5, n_features + 0.5),
        xticks=range(1, n_features + 1),
        xticklabels=feature_importance_global.index,
        xlabel=None,
        ylim=(0.5, melted.cluster.nunique() + 0.5),
        yticks=sorted(melted.cluster.unique()),
        ylabel="Cluster",
    )
    ax_plot.tick_params(axis="x", rotation=90)

    handles = [
        ax_plot.scatter([], [], s=5 + (200 - 5) * v, color="gray", alpha=0.5, label=f"{v:.1f}")
        for v in [0.1, 0.5, 1.0]
    ]

    ax_legend.axis("off")
    ax_legend.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.38, 0.5),
        ncol=1,
        fontsize=8,
        frameon=False,
        labelspacing=1.5,
        handletextpad=0.5,
        borderpad=1,
    )
    ax_legend.text(
        1.8,
        0.5,
        "Feature Importance",
        transform=ax_legend.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=8,
    )

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=CLIP_ZSCORE_NEGATIVE, vmax=CLIP_ZSCORE_POSITIVE),
            cmap=sns.color_palette(color_spec["color_features"], as_cmap=True),
        ),
        cax=ax_cbar,
        orientation="vertical",
        pad=0.1,
    )
    cbar.set_ticks(np.linspace(CLIP_ZSCORE_NEGATIVE, CLIP_ZSCORE_POSITIVE, 5))
    cbar.set_label("Feature Values (standardized)")
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()

    if save:
        save_figure(save, "_dotplot")
    if show:
        plt.show()
    else:
        return fig, axes


def plot_heatmap_classification(
    data_clustering_ranked: pd.DataFrame,
    top_n: int | None,
    heatmap_type: str,
    color_spec: dict,
    show: bool,
    save: str | None,
) -> tuple[Figure, list[Axes]] | go.Figure | None:
    """
    Plot target and feature heatmaps for classification settings, grouped by cluster order.

    The top block visualizes the encoded ``target`` row and, if present, the encoded
    ``predicted_target`` row. The bottom block visualizes standardized feature values.
    Visual gaps are inserted between clusters. ``"static"`` returns Matplotlib heatmaps,
    while ``"interactive"`` returns a Plotly figure.

    :param data_clustering_ranked: DataFrame containing ``cluster``, ``target``, optional
        ``predicted_target``, and feature columns in the desired display order.
    :type data_clustering_ranked: pd.DataFrame
    :param top_n: Number of top-ranked features shown in the title only; does not subset the input.
    :type top_n: int | None
    :param heatmap_type: Heatmap mode, either ``"static"`` or ``"interactive"``.
    :type heatmap_type: str
    :param color_spec: Color specification dictionary containing ``"color_target"``,
        ``"color_features"``, and ``"color_boundaries"``.
    :type color_spec: dict
    :param show: If ``True``, display the figure; otherwise return it.
    :type show: bool
    :param save: Base path with extension; static mode saves
        ``{stem}_heatmap{suffix}``, interactive mode saves
        ``{stem}_interactive_heatmap.html`` when provided.
    :type save: str | None

    :return: In static mode, ``(fig, list_of_axes)``; in interactive mode, ``go.Figure``;
        otherwise ``None`` when ``show`` is ``True``.
    :rtype: tuple[Figure, list[Axes]] | go.Figure | None
    """
    ### Data Processing

    cluster_labels = data_clustering_ranked["cluster"]

    target_encoder = LabelEncoder()
    rows = [target_encoder.fit_transform(data_clustering_ranked["target"])]
    idx = ["target"]
    if "predicted_target" in data_clustering_ranked.columns:
        rows.append(target_encoder.fit_transform(data_clustering_ranked["predicted_target"]))
        idx.append("predicted_target")
    target = pd.DataFrame(rows, index=idx)
    categories = target_encoder.classes_
    features = _process_features_for_plotting(data_clustering_ranked.drop(columns=idx + ["cluster"]))
    features = features.T

    # Determine cluster boundaries for separator space
    boundaries = np.where(np.diff(cluster_labels) != 0)[0] + 1
    boundaries_width = int(np.ceil(np.log(target.shape[1])))
    features, target = [_insert_boundaries(df, boundaries, boundaries_width) for df in (features, target)]

    ### Plotting
    title = f"Instance Patterns Along Decision Paths by Cluster {'(top ' + str(top_n) + 'features)' if top_n else ''}"

    if heatmap_type == "static":
        # Get plotting settings

        color_target_palette = sns.color_palette(color_spec["color_target"], n_colors=len(categories))
        color_target = ListedColormap(color_target_palette)
        color_target.set_bad(
            color=color_spec["color_boundaries"], alpha=to_rgba(color_spec["color_boundaries"])[3]
        )
        color_target_legend = {i: color_target_palette[i] for i in range(len(categories))}

        fig, ax_target, ax_target_cb, ax_features, ax_features_cb, target_plot, feature_plot = (
            _plot_heatmaps_static(
                target=target,
                features=features,
                color_target=color_target,
                color_features=color_spec["color_features"],
                color_boundaries=color_spec["color_boundaries"],
                title=title,
            )
        )

        # Add a custom legend or color bar for targets plot
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_target_legend[i], markersize=10)
            for i in range(len(categories))
        ]
        ax_target_cb.legend(
            handles,
            categories,
            title="Target Categories",
            loc="center",
            bbox_to_anchor=(0.38, 0.5),
            ncol=4,
            fontsize=8,
            frameon=False,
            labelspacing=1.5,
            handletextpad=0.5,
            borderpad=1,
        )

        plt.tight_layout()

        if save:
            save_figure(save, "_heatmap")
        if show:
            plt.show()
        else:
            return fig, [ax_target, ax_target_cb, ax_features, ax_features_cb, target_plot, feature_plot]

    elif heatmap_type == "interactive":

        color_target_palette_rgb = [
            f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
            for r, g, b in sns.color_palette(color_spec["color_target"], n_colors=len(categories))
        ]
        color_target_legend = {i: color_target_palette_rgb[i] for i in range(len(categories))}
        colorscale_target = [
            [i / (len(categories) - 1), color_target_legend[i]] for i in range(len(categories))
        ]

        fig = _plot_heatmaps_interactive(
            target=target,
            features=features,
            colorscale_target=colorscale_target,
            colorbar_target=None,
            showscale_target=False,
            colorscale_features=color_spec["color_features"],
            title=title,
        )
        for i, category in enumerate(categories):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color_target_legend[i]),
                    legendgroup=category,
                    showlegend=True,
                    name=category,
                )
            )

        if save:
            p = Path(save)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(p.parent / f"{p.stem}_interactive_heatmap.html")
        if show:
            fig.show()
        else:
            return fig

    else:
        raise ValueError(f'`heatmap_type` must be either "static" or "interactive"')


def plot_heatmap_regression(
    data_clustering_ranked: pd.DataFrame,
    top_n: int | None,
    heatmap_type: str,
    color_spec: dict,
    show: bool,
    save: str | None,
) -> tuple[Figure, list[Axes]] | go.Figure | None:
    """
    Plot target and feature heatmaps for regression settings, grouped by cluster order.

    The top block visualizes continuous ``target`` values and, if present,
    ``predicted_target`` values. The bottom block visualizes standardized feature values.
    Visual gaps are inserted between clusters. ``"static"`` returns Matplotlib heatmaps,
    while ``"interactive"`` returns a Plotly figure.

    :param data_clustering_ranked: DataFrame containing ``cluster``, ``target``, optional
        ``predicted_target``, and feature columns in the desired display order.
    :type data_clustering_ranked: pd.DataFrame
    :param top_n: Number of top-ranked features shown in the title only; does not subset the input.
    :type top_n: int | None
    :param heatmap_type: Heatmap mode, either ``"static"`` or ``"interactive"``.
    :type heatmap_type: str
    :param color_spec: Color specification dictionary containing ``"color_target"``,
        ``"color_features"``, and ``"color_boundaries"``.
    :type color_spec: dict
    :param show: If ``True``, display the figure; otherwise return it.
    :type show: bool
    :param save: Base path with extension; static mode saves
        ``{stem}_heatmap{suffix}``, interactive mode saves
        ``{stem}_interactive_heatmap.html`` when provided.
    :type save: str | None

    :return: In static mode, ``(fig, list_of_axes)``; in interactive mode, ``go.Figure``;
        otherwise ``None`` when ``show`` is ``True``.
    :rtype: tuple[Figure, list[Axes]] | go.Figure | None
    """
    ### Data Processing

    cluster_labels = data_clustering_ranked["cluster"]

    rows = [data_clustering_ranked["target"]]
    idx = ["target"]
    if "predicted_target" in data_clustering_ranked.columns:
        rows.append(data_clustering_ranked["predicted_target"])
        idx.append("predicted_target")
    target = pd.DataFrame(rows, index=idx)
    features = _process_features_for_plotting(data_clustering_ranked.drop(columns=idx + ["cluster"]))
    features = features.T

    # Determine cluster boundaries for separator space
    boundaries = np.where(np.diff(cluster_labels) != 0)[0] + 1
    boundaries_width = int(np.ceil(np.log(target.shape[1])))
    features, target = [_insert_boundaries(df, boundaries, boundaries_width) for df in (features, target)]

    ### Plotting
    title = f"Instance Patterns Along Decision Paths by Cluster {'(top ' + str(top_n) + 'features)' if top_n else ''}"

    if heatmap_type == "static":
        # Get plotting settings
        color_target_palette = sns.color_palette(color_spec["color_target"], as_cmap=True)

        # Plot heatmaps
        fig, ax_target, ax_target_cb, ax_features, ax_features_cb, target_plot, feature_plot = (
            _plot_heatmaps_static(
                target=target,
                features=features,
                color_target=color_target_palette,
                color_features=color_spec["color_features"],
                color_boundaries=color_spec["color_boundaries"],
                title=title,
            )
        )

        # Add a custom legend or color bar for targets plot
        cbar = fig.colorbar(target_plot.collections[0], ax=ax_target_cb, orientation="vertical", pad=0.1)
        cbar.set_label("Target")

        plt.tight_layout()
        if save:
            save_figure(save, "_heatmap")
        if show:
            plt.show()
        else:
            return fig, [ax_target, ax_target_cb, ax_features, ax_features_cb, target_plot, feature_plot]

    elif heatmap_type == "interactive":
        target_colorbar = dict(title="Target Scale", x=1.2)

        fig = _plot_heatmaps_interactive(
            target=target,
            features=features,
            colorscale_target=color_spec["color_target"],
            colorbar_target=dict(title="Target Scale", x=1.2),
            showscale_target=True,
            colorscale_features=color_spec["color_features"],
            title=title,
        )
        if save:
            p = Path(save)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(p.parent / f"{p.stem}_interactive_heatmap.html")
        if show:
            fig.show()
        else:
            return fig
    else:
        raise ValueError(f'`heatmap_type` must be either "static" or "interactive"')


def _process_features_for_plotting(features: pd.DataFrame) -> pd.DataFrame:
    """
    Convert feature columns into a numeric standardized representation for plotting.

    String-valued columns are encoded as categorical integer codes. All feature columns are
    then standardized using :class:`sklearn.preprocessing.StandardScaler`, returned as a
    pandas DataFrame, and clipped to the interval ``[CLIP_ZSCORE_NEGATIVE, CLIP_ZSCORE_POSITIVE]``.

    :param features: DataFrame containing only feature columns to be transformed.
    :type features: pd.DataFrame

    :return: Standardized numeric feature matrix with the same shape and index/column structure.
    :rtype: pd.DataFrame
    """

    # Encode categorical features as numeric
    for col in features.columns:
        if pd.api.types.is_string_dtype(features[col]):
            features[col] = features[col].astype("category")
            features[col] = features[col].cat.codes

    # Normalize all features (both encoded categorical and continuous)
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    features = scaler.fit_transform(features)
    features = features.clip(CLIP_ZSCORE_NEGATIVE, CLIP_ZSCORE_POSITIVE)

    return features


def _insert_boundaries(df: pd.DataFrame, boundaries: np.ndarray, boundaries_width: int) -> pd.DataFrame:
    """
    Insert NA spacer columns at cluster boundaries for visual separation in heatmaps.

    For each position in ``boundaries``, inserts ``boundaries_width`` columns filled with
    ``pd.NA`` so that adjacent clusters are separated by visible gaps in downstream plots.

    :param df: DataFrame whose columns are ordered samples and whose rows are plotted variables.
    :type df: pd.DataFrame
    :param boundaries: Column indices where a new cluster starts.
    :type boundaries: np.ndarray
    :param boundaries_width: Number of NA spacer columns inserted at each boundary.
    :type boundaries_width: int

    :return: DataFrame with additional NA columns inserted at each boundary position.
    :rtype: pd.DataFrame
    """
    add = 0
    for boundary in boundaries:
        df = pd.concat(
            [
                df.iloc[:, : boundary + add],  # until the boundary
                pd.DataFrame(  # the NA block
                    np.full((df.shape[0], boundaries_width), pd.NA),
                    index=df.index,
                ),
                df.iloc[:, boundary + add :],  # from the boundary
            ],
            axis=1,
        )
        add += boundaries_width
    return df


def _plot_heatmaps_static(
    target: pd.DataFrame,
    features: pd.DataFrame,
    color_target: ListedColormap,
    color_features: str,
    color_boundaries: str,
    title: str,
) -> tuple[Figure, Axes, Axes, Axes, Axes, Axes, Axes]:
    """
    Create the static Matplotlib layout for target and feature heatmaps.

    The figure contains a two-row layout with target heatmap on top and feature heatmap
    below, plus dedicated side axes for the target legend/color bar and the feature color
    bar. Missing values introduced as cluster separators are masked during plotting.

    :param target: Target matrix with rows such as ``target`` and ``predicted_target`` and
        columns representing ordered samples, including NA separator columns.
    :type target: pd.DataFrame
    :param features: Feature matrix with rows representing features and columns aligned to
        ``target``, including NA separator columns.
    :type features: pd.DataFrame
    :param color_target: Colormap used for the target heatmap.
    :type color_target: ListedColormap
    :param color_features: Name of the seaborn/Matplotlib colormap used for the feature heatmap.
    :type color_features: str
    :param color_boundaries: Color used for masked separator cells in the feature heatmap.
    :type color_boundaries: str
    :param title: Figure title.
    :type title: str

    :return: Figure, target axes, target-side axes, feature axes, feature-side axes,
        target heatmap artist, and feature heatmap artist.
    :rtype: tuple[Figure, Axes, Axes, Axes, Axes, Axes, Axes]
    """

    # Set up the figure and subplots
    sns.set_theme(style="white", context="paper")

    figure_height = max(6.5, int(np.ceil(5 * len(features) / 25)))
    figure_width = 2.5 * figure_height

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        sharex=True,
        height_ratios=[1, 9],
        width_ratios=[20, 1],
        figsize=(figure_width, figure_height),
        facecolor="none",
    )
    fig.suptitle(title, fontsize=14)

    # Add extra axis for legends and disable plot in this axis
    ax_target, ax_target_cb = axes[0]
    ax_features, ax_features_cb = axes[1]

    ax_target_cb.axis("off")
    ax_features_cb.axis("off")

    # Adjust the axes
    ax_target.set_xticks([])
    ax_features.set_xlabel("Samples")
    ax_features.set_xticks([])

    def nas_to_min_numeric(na_df: pd.DataFrame) -> pd.DataFrame:
        t = na_df.apply(pd.to_numeric, errors="coerce")
        t = t.fillna(t.min(skipna=True))
        return t

    # Plot the target heatmap
    target_plot = sns.heatmap(
        nas_to_min_numeric(target),
        mask=target.isna(),
        ax=ax_target,
        cmap=color_target,
        cbar=False,
        yticklabels=target.index,
        xticklabels=False,
    )
    ax_target.set_yticklabels(ax_target.get_yticklabels(), rotation=0)

    features_cmap = sns.color_palette(color_features, as_cmap=True)
    features_cmap.set_bad(color=color_boundaries, alpha=to_rgba(color_boundaries)[3])

    # Plot the feature heatmap
    feature_plot = sns.heatmap(
        nas_to_min_numeric(features),
        mask=features.isna(),
        ax=ax_features,
        cmap=features_cmap,
        cbar=False,
        yticklabels=features.index,
        xticklabels=False,
    )
    # Add color bar feature plot
    cbar = fig.colorbar(feature_plot.collections[0], ax=ax_features_cb, orientation="vertical", pad=0.1)
    cbar.set_label("Features Values (standardized)")

    return fig, ax_target, ax_target_cb, ax_features, ax_features_cb, target_plot, feature_plot


def _plot_heatmaps_interactive(
    target: pd.DataFrame,
    features: pd.DataFrame,
    colorscale_target: list,
    colorbar_target: dict,
    showscale_target: bool,
    colorscale_features: str,
    title: str,
) -> go.Figure:
    """
    Create the interactive Plotly layout for target and feature heatmaps.

    The figure contains two vertically stacked heatmaps with shared x-axis: a target block
    on top and a feature block below. The target heatmap uses the provided target color
    scale, while the feature heatmap converts a Matplotlib colormap to a Plotly colorscale.

    :param target: Target matrix with rows such as ``target`` and ``predicted_target`` and
        columns representing ordered samples, including NA separator columns.
    :type target: pd.DataFrame
    :param features: Feature matrix with rows representing features and columns aligned to
        ``target``.
    :type features: pd.DataFrame
    :param colorscale_target: Plotly colorscale definition for the target heatmap.
    :type colorscale_target: list
    :param colorbar_target: Color bar configuration for the target heatmap.
    :type colorbar_target: dict
    :param showscale_target: Whether to display the target heatmap color bar.
    :type showscale_target: bool
    :param colorscale_features: Name of the Matplotlib colormap converted for the feature heatmap.
    :type colorscale_features: str
    :param title: Figure title.
    :type title: str

    :return: Configured interactive heatmap figure.
    :rtype: go.Figure
    """

    fig = make_subplots(rows=2, cols=1, row_heights=[0.15, 0.85], shared_xaxes=True, vertical_spacing=0.02)

    # Target heatmap (top row) with categorical colors
    fig.add_trace(
        go.Heatmap(
            z=target.values,
            y=target.index,
            colorscale=colorscale_target,
            colorbar=colorbar_target,
            showscale=showscale_target,
            hoverongaps=False,
            customdata=np.tile(np.array(features.columns, dtype=str), target.shape),
            hovertemplate="<b>Sample: %{customdata}</b><br>" + "x: %{x}<br>" + "target: %{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Reverse order the rows of the dataframe to plot in correct order
    features = features.iloc[::-1]

    # Continuous feature heatmap
    fig.add_trace(
        go.Heatmap(
            z=features.values,
            y=features.index,
            colorscale=matplotlib_to_plotly(colorscale_features),
            colorbar=dict(title="Features", x=1.1),  # Adjust position of color bar
            showscale=True,
            hoverongaps=False,
            customdata=np.tile(np.array(features.columns, dtype=str), (features.shape[0], 1)),
            hovertemplate="<b>Sample: %{customdata}</b><br>"
            + "x: %{x}<br>"
            + "y: %{y}<br>"
            + "value: %{z}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Add axis title and layout adjustments
    fig.add_annotation(
        x=0.5,
        y=-0.2,
        showarrow=False,
        text="Samples",
        xref="paper",
        yref="paper",
        font=dict(size=16),
    )
    fig.update_layout(
        title=dict(text=title, y=0.95, x=0.5, xanchor="center", yanchor="top"),
        legend=dict(yanchor="top", y=0.9, xanchor="right", x=1.3),
        plot_bgcolor="rgba(0,0,0,0)",
        margin_pad=5,  # add some space between the row labels and the heatmap itself
    )
    # make sure ALL row labels are shown
    row_labels = (
        ["target"]
        + (["predicted_target"] if "predicted_target" in target.index else [])
        + list(features.index.astype(str))
    )
    fig.update_yaxes(
        type="category",
        tickmode="array",
        ticktext=row_labels,
        tickvals=row_labels,
    )

    return fig
