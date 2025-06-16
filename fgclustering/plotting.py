############################################
# Imports
############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from .utils import matplotlib_to_plotly

############################################
# Plotting Functions
############################################


def _plot_feature_importance(
    feature_importance_global: pd.Series,
    feature_importance_local: pd.DataFrame,
    top_n: int,
    num_cols: int,
    save: str,
):

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


def _plot_distributions(
    data_clustering_ranked: pd.DataFrame,
    top_n: int,
    num_cols: int,
    cmap_target_dict: dict,
    save: str,
):

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
                orientation="vertical",
            )
            ax.set_title(f"Feature: {feature}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        plt.savefig(f"{save}_boxplots.png", bbox_inches="tight", dpi=300)
    else:
        plt.show()


def _plot_heatmap_classification(
    data_clustering_ranked: pd.DataFrame,
    top_n: int,
    heatmap_type: str,
    cmap_target_dict: dict,
    save: str,
):

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
            cmap_target = sns.color_palette(color_palette, as_cmap=True)
        else:
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
        else:
            plt.show()

    elif heatmap_type == "interactive":
        if cmap_target_dict is not None:
            # Create a color scale for the target categories
            color_palette_rgb = [
                f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
                for r, g, b in sns.color_palette(list(cmap_target_dict.values()))
            ]
        else:
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

        if save:
            fig.write_html(f"{save}_heatmap.html")
        else:
            fig.show()


def _plot_heatmap_regression(
    data_clustering_ranked: pd.DataFrame,
    top_n: int,
    heatmap_type: str,
    save: str,
):

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


def _process_features_for_heatmap(features):

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


def _get_heatmap_plotting_settings(target, top_n):

    color_target = "Greens"
    color_features = "coolwarm"
    boundaries_color = "white"

    boundaries_width = int(np.ceil(np.log(target.shape[1])))

    title = f"Subgroups of instances that follow similar decision paths in the RF model \n Showing {'top ' + str(top_n) if top_n else 'all'} features"

    return color_target, color_features, boundaries_color, boundaries_width, title


def _plot_heatmaps_static(
    target, cmap_target, features, features_color, boundaries, boundaries_color, boundaries_width, title
):

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
