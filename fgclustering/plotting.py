############################################
# Imports
############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from typing import Tuple, Any
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from .utils import matplotlib_to_plotly

############################################
# Plotting Functions
############################################


def plot_feature_importance(
    feature_importance_global: pd.Series,
    feature_importance_local: pd.DataFrame,
    top_n: int,
    num_cols: int,
    save: str,
) -> None:
    """
    Visualize global and local feature importance values as bar charts.

    :param feature_importance_local: Local importance values per cluster.
    :type feature_importance_local: pandas.DataFrame
    :param feature_importance_global: Global mean importance values across clusters.
    :type feature_importance_global: pandas.Series
    :param top_n: If specified, number of top-ranked features to plot.
    :type top_n: int
    :param num_cols: Number of columns in the subplot layout.
    :type num_cols: int
    :param save: If specified, path prefix to save plots.
    :type save: int
    """
    # Determine figure size dynamically based on the number of features
    num_features = len(feature_importance_global.index)
    figsize_width = 6.5
    figsize_height = max(figsize_width, int(np.ceil(5 * num_features / 25)))

    num_subplots = 1 + feature_importance_local.shape[1]
    num_cols = min(num_cols, num_subplots)
    num_rows = int(np.ceil(num_subplots / num_cols))

    plt.figure(figsize=(num_cols * figsize_width, num_rows * figsize_height))
    plt.subplots_adjust(top=0.95, hspace=0.8, wspace=0.8)
    plt.suptitle(
        f"Feature Importance - Showing {'top ' + str(top_n) if top_n else 'all'} features",
        fontsize=14,
    )
    sns.set_theme(style="whitegrid")

    # Plot global feature importance
    importance_global = pd.DataFrame(
        {
            "Feature": feature_importance_global.index,
            "Importance": feature_importance_global.to_list(),
        }
    ).sort_values(by="Importance", ascending=False)

    ax = plt.subplot(num_rows, num_cols, 1)
    sns.barplot(data=importance_global, x="Importance", y="Feature", color="#3470a3")
    ax.set_title(f"Cluster all")

    # Plot local feature importance
    for n, cluster in enumerate(feature_importance_local.columns):
        importance_local = pd.DataFrame(
            {
                "Feature": feature_importance_local.index,
                "Importance": feature_importance_local[cluster].to_list(),
            }
        ).sort_values(by="Importance", ascending=False)
        ax = plt.subplot(num_rows, num_cols, n + 2)
        sns.barplot(data=importance_local, x="Importance", y="Feature", color="#3470a3")
        ax.set_title(f"Cluster {cluster}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        plt.savefig(f"{save}_feature_importance.png", bbox_inches="tight", dpi=300)
    else:
        plt.show()


def plot_distributions(
    data_clustering_ranked: pd.DataFrame,
    top_n: int,
    num_cols: int,
    cmap_target_dict: dict,
    save: str,
) -> None:
    """
    Plot the decision patterns that emerge from forest-guided clustering using feature distribution plots.

    :param data_clustering: DataFrame of clustering data ordered by the global feature importance.
    :type data_clustering: pd.DataFrame
    :param top_n: If specified, number of top-ranked features to plot.
    :type top_n: int
    :param num_cols: Number of columns in the subplot layout.
    :type num_cols: int
    :param cmap_target_dict: If specified, custom color map for categorical targets.
    :type cmap_target_dict: dict
    :param save: If specified, path prefix to save plots.
    :type save: str
    """

    features_to_plot = data_clustering_ranked.drop("cluster", axis=1, inplace=False).columns.to_list()

    num_rows = int(np.ceil(len(features_to_plot) / num_cols))

    plt.figure(figsize=(num_cols * 4.5, num_rows * 4.5))
    plt.subplots_adjust(top=0.95, hspace=0.8, wspace=0.8)
    plt.suptitle(
        f"Distribution of feature values across subgroups - Showing {'top ' + str(top_n) if top_n else 'all'} features",
        fontsize=14,
    )

    if cmap_target_dict is not None:
        color_palette = sns.color_palette(list(cmap_target_dict.values()))
        cmap_target = sns.color_palette(color_palette, as_cmap=True)
    else:
        cmap_target = "Blues_r"

    for n, feature in enumerate(features_to_plot):
        # add a new subplot iteratively
        ax = plt.subplot(num_rows, num_cols, n + 1)
        if data_clustering_ranked[feature].nunique() < 5 or isinstance(
            data_clustering_ranked[feature].dtype, pd.CategoricalDtype
        ):
            sns.countplot(
                x="cluster",
                hue=feature,
                data=data_clustering_ranked,
                ax=ax,
                palette=cmap_target,
            )
            ax.set_title(f"Feature: {feature}")
            ax.legend(bbox_to_anchor=(1, 1), loc=2)
        else:
            sns.boxplot(
                x="cluster",
                y=feature,
                data=data_clustering_ranked,
                ax=ax,
                color="#3470a3",
                orient="v",
            )
            ax.set_title(f"Feature: {feature}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        plt.savefig(f"{save}_boxplots.png", bbox_inches="tight", dpi=300)
    else:
        plt.show()


def plot_heatmap_classification(
    data_clustering_ranked: pd.DataFrame,
    top_n: int,
    heatmap_type: str,
    cmap_target_dict: dict,
    save: str,
) -> None:
    """
    Plot the decision patterns that emerge from forest-guided clustering using feature heatmaps for classification tasks.

    :param data_clustering: DataFrame of clustering data ordered by the global feature importance.
    :type data_clustering: pd.DataFrame
    :param top_n: If specified, number of top-ranked features to plot.
    :type top_n: int
    :param heatmap_type: Heatmap shown in "static" or "interactive" style.
    :type heatmap_type: str
    :param cmap_target_dict: If specified, custom color map for categorical targets.
    :type cmap_target_dict: dict
    :param save: If specified, path prefix to save plots.
    :type save: str
    """
    cluster_labels = data_clustering_ranked["cluster"]

    # Encode categorical target for plotting and transpose for plotting
    target_encoder = LabelEncoder()
    target = target_encoder.fit_transform(data_clustering_ranked["target"])
    target = pd.DataFrame([target], index=["target"])

    categories = target_encoder.classes_

    # Process features and transpose for plotting
    features = _process_features_for_heatmap(data_clustering_ranked.drop(columns=["target", "cluster"]))
    features = features.T

    # Determine cluster boundaries for separator lines
    boundaries = np.where(np.diff(cluster_labels) != 0)[0] + 1

    target_color, features_color, boundaries_color, boundaries_width, title = _get_heatmap_plotting_settings(
        target, top_n
    )

    if heatmap_type == "static":
        # Get plotting settings

        if cmap_target_dict is not None:
            color_palette = sns.color_palette(list(cmap_target_dict.values()))
            target_cmap = sns.color_palette(color_palette, as_cmap=True)
        else:
            color_palette = sns.color_palette(target_color, n_colors=len(categories))
            target_cmap = sns.color_palette(color_palette, as_cmap=True)

        color_map = {i: color_palette[i] for i in range(len(categories))}

        fig, ax_target, ax_target_cb, ax_features, ax_features_cb, target_plot, feature_plot = (
            _plot_heatmaps_static(
                target,
                target_cmap,
                features,
                features_color,
                boundaries,
                boundaries_color,
                boundaries_width,
                title,
            )
        )

        # Add a custom legend or color bar for targets plot
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[i], markersize=10)
            for i in range(len(categories))
        ]
        ax_features.legend(
            handles,
            categories,
            title="Target Categories",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=len(categories),
        )

        plt.tight_layout()
        if save:
            plt.savefig(f"{save}_heatmap.png", bbox_inches="tight", dpi=300)
        else:
            plt.show()

    elif heatmap_type == "interactive":
        if cmap_target_dict is not None:
            # Create a color scale for the target categories
            target_color_palette_rgb = [
                f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
                for r, g, b in sns.color_palette(list(cmap_target_dict.values()))
            ]
        else:
            # Create a color scale for the target categories
            target_color_palette_rgb = [
                f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
                for r, g, b in sns.color_palette(target_color, n_colors=len(categories))
            ]

        target_color_map = {i: target_color_palette_rgb[i] for i in range(len(categories))}
        target_colorscale = [[i / (len(categories) - 1), target_color_map[i]] for i in range(len(categories))]

        fig = _plot_heatmaps_interactive(
            target,
            target_colorscale,
            None,
            False,
            features,
            features_color,
            boundaries,
            boundaries_color,
            boundaries_width,
            title,
        )
        for i, category in enumerate(categories):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=target_color_map[i]),
                    legendgroup=category,
                    showlegend=True,
                    name=category,
                )
            )

        if save:
            fig.write_html(f"{save}_heatmap.html")
        else:
            fig.show()


def plot_heatmap_regression(
    data_clustering_ranked: pd.DataFrame,
    top_n: int,
    heatmap_type: str,
    save: str,
) -> None:
    """
    Plot the decision patterns that emerge from forest-guided clustering using feature heatmaps for regression tasks.

    :param data_clustering: DataFrame of clustering data ordered by the global feature importance.
    :type data_clustering: pd.DataFrame
    :param top_n: If specified, number of top-ranked features to plot.
    :type top_n: int
    :param heatmap_type: Heatmap shown in "static" or "interactive" style.
    :type heatmap_type: str
    :param save: If specified, path prefix to save plots.
    :type save: str
    """
    cluster_labels = data_clustering_ranked["cluster"]

    # Get traget and transpose for plotting
    target = data_clustering_ranked["target"]
    target = pd.DataFrame([target], index=["target"])

    # Process features and transpose for plotting
    features = _process_features_for_heatmap(data_clustering_ranked.drop(columns=["target", "cluster"]))
    features = features.T

    # Determine cluster boundaries for separator lines
    boundaries = np.where(np.diff(cluster_labels) != 0)[0] + 1
    target_color, features_color, boundaries_color, boundaries_width, title = _get_heatmap_plotting_settings(
        target, top_n
    )

    if heatmap_type == "static":
        # Get plotting settings
        cmap_target = sns.color_palette(target_color, as_cmap=True)

        # Plot heatmaps
        fig, ax_target, ax_target_cb, ax_features, ax_features_cb, target_plot, feature_plot = (
            _plot_heatmaps_static(
                target,
                cmap_target,
                features,
                features_color,
                boundaries,
                boundaries_color,
                boundaries_width,
                title,
            )
        )

        # Add a custom legend or color bar for targets plot
        cbar = fig.colorbar(target_plot.collections[0], ax=ax_target_cb, orientation="vertical", pad=0.1)
        cbar.set_label("Target")

        plt.tight_layout()
        if save:
            plt.savefig(f"{save}_heatmap.png", bbox_inches="tight", dpi=300)
        plt.show()

    elif heatmap_type == "interactive":
        target_colorbar = dict(title="Target Scale", x=1.2)

        fig = _plot_heatmaps_interactive(
            target,
            target_color,
            target_colorbar,
            True,
            features,
            features_color,
            boundaries,
            boundaries_color,
            boundaries_width,
            title,
        )
        fig.show()

        if save:
            raise RuntimeError(
                "Saving interactive plots is not implemented. Please set heatmap_type='static' to save the plot."
            )


def _process_features_for_heatmap(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert categorical features to numeric codes and normalize all features using MinMax scaling.

    :param features: DataFrame containing the features to be encoded and scaled.
    :type features: pandas.DataFrame

    :return: Transformed and normalized features.
    :rtype: pandas.DataFrame
    """

    # Encode categorical features as numeric
    for col in features.columns:
        if pd.api.types.is_string_dtype(features[col]):
            features[col] = features[col].astype("category")
            features[col] = features[col].cat.codes

    # Normalize all features (both encoded categorical and continuous)
    scaler = MinMaxScaler()
    scaler.set_output(transform="pandas")
    features = scaler.fit_transform(features)

    return features


def _get_heatmap_plotting_settings(
    target: pd.Series,
    top_n: int,
) -> Tuple[str, str, str, int, str]:
    """
    Define color schemes, boundary widths, and titles for heatmap plotting.

    :param target: Target variable.
    :type target: pandas.Series
    :param top_n: If specified, number of top-ranked features to display in the title.
    :type top_n: int

    :return: Tuple with color settings, boundary width, and heatmap title.
    :rtype: Tuple[str, str, str, int, str]
    """

    color_target = "Greens"
    color_features = "coolwarm"
    boundaries_color = "white"

    boundaries_width = int(np.ceil(np.log(target.shape[1])))

    title = f"Subgroups of instances that follow similar decision paths in the RF model \n Showing {'top ' + str(top_n) if top_n else 'all'} features"

    return color_target, color_features, boundaries_color, boundaries_width, title


def _plot_heatmaps_static(
    target: pd.Series,
    target_cmap: dict,
    features: pd.DataFrame,
    features_color: str,
    boundaries: np.ndarray,
    boundaries_color: str,
    boundaries_width: int,
    title: str,
) -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
    """
    Create a static (matplotlib) heatmap of target and feature values with visual cluster boundaries.

    :param target: Target variable.
    :type target: pandas.Series
    :param target_cmap: Colormap used for the target heatmap.
    :type target_cmap: dict
    :param features: Normalized feature matrix.
    :type features: pandas.DataFrame
    :param features_color: Colormap used for the feature heatmap.
    :type features_color: str
    :param boundaries: Positions for vertical cluster boundary lines.
    :type boundaries: np.ndarray
    :param boundaries_color: Color used for boundary lines.
    :type boundaries_color: str
    :param boundaries_width: Line width used for boundary lines.
    :type boundaries_width: int
    :param title: Title of the heatmap figure.
    :type title: str

    :return: Tuple of figure and axes objects for both heatmaps and colorbars.
    :rtype: Tuple[Any, Any, Any, Any, Any, Any, Any]
    """

    # Set up the figure and subplots
    figure_size = max(6.5, int(np.ceil(5 * len(features) / 25)))
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        sharex=True,
        height_ratios=[1, 9],
        width_ratios=[20, 1],
        figsize=(2.5 * figure_size, figure_size),
    )

    # Add extra axis for legends and disable plot in this axis
    ax_target, ax_target_cb = axes[0]
    ax_features, ax_features_cb = axes[1]

    ax_target_cb.axis("off")
    ax_features_cb.axis("off")

    # Adjust the axes
    ax_target.set_title(title)
    ax_target.set_xticks([])

    ax_features.set_xlabel("Samples")
    ax_features.set_xticks([])

    # Plot the target heatmap
    target_plot = sns.heatmap(
        target,
        ax=ax_target,
        cmap=target_cmap,
        cbar=False,
        yticklabels=target.index,
        xticklabels=False,
    )
    ax_target.set_yticklabels(ax_target.get_yticklabels(), rotation=0)

    # Plot the feature heatmap
    feature_plot = sns.heatmap(
        features,
        ax=ax_features,
        cmap=features_color,
        cbar=False,
        yticklabels=features.index,
        xticklabels=False,
    )
    # Add color bar feature plot
    cbar = fig.colorbar(feature_plot.collections[0], ax=ax_features_cb, orientation="vertical", pad=0.1)
    cbar.set_label("Features")

    # Add cluster separators
    for ax in [ax_target, ax_features]:
        for boundary in boundaries:
            ax.axvline(boundary, color=boundaries_color, lw=boundaries_width)

    return fig, ax_target, ax_target_cb, ax_features, ax_features_cb, target_plot, feature_plot


def _plot_heatmaps_interactive(
    target: pd.Series,
    target_colorscale: list,
    target_colorbar: dict,
    target_showscale: bool,
    features: pd.DataFrame,
    features_color: str,
    boundaries: np.ndarray,
    boundaries_color: str,
    boundaries_width: int,
    title: str,
) -> go.Figure:
    """
    Create an interactive (Plotly) heatmap of target and feature values with visual cluster boundaries.

    :param target: Target variable.
    :type target: pandas.Series
    :param target_colorscale: Color scale used for the target heatmap.
    :type target_colorscale: list
    :param target_colorbar: Color bar used for the target heatmap.
    :type target_colorbar: dict
    :param target_showscale: Whether to display the color scale for the target heatmap.
    :type target_showscale: bool
    :param features: Normalized feature matrix.
    :type features: pandas.DataFrame
    :param features_color: Colormap used for the feature heatmap.
    :type features_color: str
    :param boundaries: Positions for vertical cluster boundary lines.
    :type boundaries: np.ndarray
    :param boundaries_color: Color used for boundary lines.
    :type boundaries_color: str
    :param boundaries_width: Line width used for boundary lines.
    :type boundaries_width: int
    :param title: Title of the heatmap figure.
    :type title: str

    :return: Plotly figure object containing the interactive heatmap.
    :rtype: go.Figure
    """

    fig = make_subplots(rows=2, cols=1, row_heights=[0.1, 0.9], shared_xaxes=True, vertical_spacing=0.02)

    # Target heatmap (top row) with categorical colors
    fig.add_trace(
        go.Heatmap(
            z=target.values,  # Use only the target values
            y=target.index,
            colorscale=target_colorscale,
            colorbar=target_colorbar,
            showscale=target_showscale,  # Hide color bar for the target
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
            colorscale=matplotlib_to_plotly(features_color),
            colorbar=dict(title="Features", x=1.1),  # Adjust position of color bar
            showscale=True,
        ),
        row=2,
        col=1,
    )

    # Add separators for clusters
    for xref, yref, y1 in [("x1", "y1", 0.5), ("x2", "y2", len(features.index) - 0.5)]:
        for boundary in boundaries:
            fig.add_shape(
                type="line",
                x0=boundary - 0.5,
                x1=boundary - 0.5,
                y0=-0.5,
                y1=y1,
                line=dict(color=boundaries_color, width=boundaries_width),
                xref=xref,
                yref=yref,
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
    )

    return fig
