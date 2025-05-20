############################################
# imports
############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import fgclustering.utils as utils


############################################
# functions
############################################


def _plot_feature_importance(
    distance_of_features_ranked: pd.DataFrame,
    distance_of_features_per_cluster: pd.DataFrame,
    thr_distance: float,
    top_n: int,
    num_cols: int,
    save: str,
):
    """
    Generate and display a plot showing the importance of features based on distances.
    The plot includes both global feature importance and local feature importance for each cluster.
    Global importance is based on all clusters combined, while local importance is specific to each cluster.

    :param distance_of_features_ranked: DataFrame containing distances of features, ranked by distance.
    :type distance_of_features_ranked: pandas.DataFrame
    :param distance_of_features_per_cluster: DataFrame containing distances of features for each cluster.
    :type distance_of_features_per_cluster: pandas.DataFrame
    :param thr_distance: Distance threshold for display. Only features with distances above this threshold
                    are considered. Defaults to 1 (no filtering).
    :type thr_distance: float, optional
    :param top_n: Number of top features to display in the plot. If None, all features are included.
                Defaults to None.
    :type top_n: int, optional
    :param num_cols: Number of plots per row in the output figure. Defaults to 4.
    :type num_cols: int, optional
    :param save: Filename to save the plot. If None, the plot will not be saved. Defaults to None.
    :type save: str, optional
    """

    # Determine figure size dynamically based on the number of features
    num_features = len(distance_of_features_ranked.columns)
    figsize_width = 6.5
    figsize_height = max(figsize_width, int(np.ceil(5 * num_features / 25)))

    num_subplots = 1 + distance_of_features_per_cluster.shape[1]
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
            "Feature": distance_of_features_ranked.columns,
            "Importance": distance_of_features_ranked.to_list(),
        }
    ).sort_values(by="Importance", ascending=False)

    ax = plt.subplot(num_rows, num_cols, 1)
    sns.barplot(data=importance_global, x="Importance", y="Feature", color="#3470a3")
    ax.axvline(
        x=thr_distance, color="red", linestyle="--", label=f"thr distance = {thr_distance}"
    )
    ax.set_title(f"Cluster all")
    ax.legend(bbox_to_anchor=(1, 1), loc=2)

    # Plot local feature importance
    for n, cluster in enumerate(distance_of_features_per_cluster.columns):
        importance_local = pd.DataFrame(
            {
                "Feature": distance_of_features_per_cluster.index,
                "Importance": distance_of_features_per_cluster[cluster].to_list(),
            }
        ).sort_values(by="Importance", ascending=False)
        ax = plt.subplot(num_rows, num_cols, n + 2)
        sns.barplot(data=importance_local, x="Importance", y="Feature", color="#3470a3")
        ax.axvline(x=thr_distance, color="red", linestyle="--")
        ax.set_title(f"Cluster {cluster}")
        # ax.legend(bbox_to_anchor=(1, 1), loc=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        plt.savefig(f"{save}_feature_importance.png", bbox_inches="tight", dpi=300)
    plt.show()


def _plot_distributions(
    data_clustering_ranked: pd.DataFrame, thr_distance: float, top_n: int, num_cols: int, save: str
):
    """
    Plot feature boxplots (for continuous features) or barplots (for categorical features) divided by clusters,
    where features are filtered and ranked by distribution distances.

    :param data_clustering_ranked: Filtered and ranked data frame incl features, target and cluster numbers.
    :type data_clustering_ranked: pandas.DataFrame
    :param thr_distance: Distance threshold used for feature filtering
    :type thr_distance: float, optional
    :param num_cols: Number of plots in one row.
    :type num_cols: int
    :param save: Filename to save plot.
    :type save: str
    """
    features_to_plot = data_clustering_ranked.drop("cluster", axis=1, inplace=False).columns.to_list()

    num_rows = int(np.ceil(len(features_to_plot) / num_cols))

    plt.figure(figsize=(num_cols * 4.5, num_rows * 4.5))
    plt.subplots_adjust(top=0.95, hspace=0.8, wspace=0.8)
    plt.suptitle(
        f"Distribution of feature values across subgroups - Showing {'top ' + str(top_n) if top_n else 'all'} features with distance > {thr_distance}",
        fontsize=14,
    )

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
                palette="Blues_r",
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
            )
            ax.set_title(f"Feature: {feature}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        plt.savefig(f"{save}_boxplots.png", bbox_inches="tight", dpi=300)
    plt.show()


def _plot_heatmap_classification(
    data_clustering_ranked: pd.DataFrame,
    thr_distance: float,
    top_n: int,
    heatmap_type: str,
    save: str,
):
    """
    Generates a classification heatmap visualization for clustered data, supporting both static and interactive plots.
    Displays target and feature heatmaps with cluster boundaries.

    :param data_clustering_ranked: A DataFrame containing the data for clustering, including target and cluster labels.
    :type data_clustering_ranked: pd.DataFrame
    :param thr_distance: Threshold for feature distance in the heatmap.
    :type thr_distance: float
    :param top_n: Number of top features to display in the heatmap, or None to display all features.
    :type top_n: int
    :param heatmap_type: Type of heatmap to generate: "static" for Matplotlib or "interactive" for Plotly.
    :type heatmap_type: str
    :param save: File path for saving the heatmap. Only supported for static heatmaps.
    :type save: str
    :raises RuntimeError: If `heatmap_type` is "interactive" and `save` is specified, as saving interactive plots is not implemented.
    :return: None
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
        target, top_n, thr_distance
    )

    if heatmap_type == "static":
        # Get plotting settings

        color_palette = sns.color_palette(target_color, n_colors=len(categories))
        cmap_target = sns.color_palette(color_palette, as_cmap=True)
        color_map = {i: color_palette[i] for i in range(len(categories))}

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
        plt.show()

    elif heatmap_type == "interactive":
        # Create a color scale for the target categories
        color_palette_rgb = [
            f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
            for r, g, b in sns.color_palette(target_color, n_colors=len(categories))
        ]
        color_map = {i: color_palette_rgb[i] for i in range(len(categories))}
        colorscale = [[i / (len(categories) - 1), color_map[i]] for i in range(len(categories))]

        fig = _plot_heatmaps_interactive(
            target,
            colorscale,
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
                    marker=dict(size=10, color=color_map[i]),
                    legendgroup=category,
                    showlegend=True,
                    name=category,
                )
            )
        fig.show()

        if save:
            raise RuntimeError(
                "Saving interactive plots is not implemented. Please set heatmap_type='static' to save the plot."
            )


def _plot_heatmap_regression(
    data_clustering_ranked: pd.DataFrame,
    thr_distance: float,
    top_n: int,
    heatmap_type: str,
    save: str,
):
    """
    Generates a regression heatmap visualization for clustered data, with options for static or interactive plots.
    Displays target and feature heatmaps with cluster boundaries.

    :param data_clustering_ranked: A DataFrame containing the data for clustering, including target and cluster labels.
    :type data_clustering_ranked: pd.DataFrame
    :param thr_distance: Threshold for feature distance in the heatmap.
    :type thr_distance: float
    :param top_n: Number of top features to display in the heatmap, or None to display all features.
    :type top_n: int
    :param heatmap_type: Type of heatmap to generate: "static" for Matplotlib or "interactive" for Plotly.
    :type heatmap_type: str
    :param save: File path for saving the heatmap. Only supported for static heatmaps.
    :type save: str
    :raises RuntimeError: If `heatmap_type` is "interactive" and `save` is specified, as saving interactive plots is not implemented.
    :return: None
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
        target, top_n, thr_distance
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


def _process_features_for_heatmap(features):
    """
    Processes a DataFrame of features to prepare them for heatmap visualization.
    Categorical features are encoded as numeric, and all features are normalized to a 0-1 range.

    :param features: A DataFrame containing the features to process.
    :type features: pd.DataFrame
    :return: A DataFrame with categorical features encoded as numeric and all features normalized.
    :rtype: pd.DataFrame
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


def _get_heatmap_plotting_settings(target, top_n, thr_distance):
    """
    Configures plotting settings for heatmap visualizations, including color schemes,
    boundary thickness, and plot title.

    :param target: Target data, i.e. a DataFrame containing the target values.
    :type target: pd.DataFrame
    :param top_n: Number of top features to display, or None to display all features.
    :type top_n: int or None
    :param thr_distance: Threshold for distance in feature selection.
    :type thr_distance: float
    :return: Tuple containing color settings for target and features, boundary color, boundary length, and plot title.
    :rtype: tuple(str, str, str, int, str)
    """
    color_target = "Greens"
    color_features = "coolwarm"
    boundaries_color = "white"

    boundaries_width = int(np.ceil(np.log(target.shape[1])))

    title = f"Subgroups of instances that follow similar decision paths in the RF model \n Showing {'top ' + str(top_n) if top_n else 'all'} features with distance > {thr_distance}"

    return color_target, color_features, boundaries_color, boundaries_width, title


def _plot_heatmaps_static(
    target, cmap_target, features, features_color, boundaries, boundaries_color, boundaries_width, title
):
    """
    Creates static heatmaps for target and features with cluster boundaries.

    :param target: DataFrame representing the target values to plot.
    :type target: pd.DataFrame
    :param cmap_target: Colormap for the target heatmap.
    :type cmap_target: str or matplotlib.colors.Colormap
    :param features: DataFrame representing the feature values to plot.
    :type features: pd.DataFrame
    :param features_color: Colormap for the feature heatmap.
    :type features_color: str or matplotlib.colors.Colormap
    :param boundaries: List of boundary positions for separating clusters in the heatmaps.
    :type boundaries: list of int
    :param boundaries_color: Color of the boundary lines.
    :type boundaries_color: str
    :param boundaries_width: Width of the boundary lines.
    :type boundaries_width: float
    :param title: Title for the heatmap plot.
    :type title: str
    :return: Tuple containing the figure, target axis, target color bar axis, features axis, features color bar axis,
             target heatmap plot, and features heatmap plot.
    :rtype: tuple(matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes, matplotlib.axes.Axes,
                  matplotlib.axes.Axes, sns.heatmap, sns.heatmap)
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
        cmap=cmap_target,
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
    target,
    target_colorscale,
    target_colorbar,
    target_showscale,
    features,
    features_color,
    boundaries,
    boundaries_color,
    boundaries_width,
    title,
):
    """
    Create an interactive Plotly heatmap visualization for target and features.

    :param target: Target data as a DataFrame.
    :param target_colorscale: Colorscale for the target heatmap.
    :param target_colorbar: Configuration for the target heatmap colorbar.
    :param target_showscale: Whether to display the color scale for the target heatmap.
    :param features: Features data as a DataFrame.
    :param features_color: Colorscale for the features heatmap.
    :param boundaries: List of boundary positions for separators.
    :param boundaries_color: Color of the boundary lines.
    :param boundaries_width: Width of the boundary lines.
    :param title: Title of the heatmap plot.
    :return: A Plotly figure object.
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
            colorscale=utils.matplotlib_to_plotly(features_color),
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
