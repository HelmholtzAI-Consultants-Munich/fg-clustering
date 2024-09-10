############################################
# imports
############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import fgclustering.utils as utils


############################################
# functions
############################################


def log_transform(p_values: list, epsilon: float = 1e-50):
    """
    Apply a log transformation to p-values to enhance numerical stability and highlight differences.
    Adds a small constant `epsilon` to avoid taking the log of zero and normalizes by dividing by
    the log of `epsilon`.

    :param p_values: List of p-values to be transformed.
    :type p_values: list
    :param epsilon: Small constant added to p-values to avoid log of zero. Defaults to 1e-50.
    :type epsilon: float, optional
    :return: Transformed p-values after log transformation.
    :rtype: numpy.ndarray
    """
    # add a small constant epsilon
    p_values = np.clip(p_values, epsilon, 1)
    return -np.log(p_values) / -np.log(epsilon)


def _plot_feature_importance(
    p_value_of_features_ranked: pd.DataFrame,
    p_value_of_features_per_cluster: pd.DataFrame,
    thr_pvalue: float,
    top_n: int,
    num_cols: int,
    save: str,
):
    """
    Generate and display a plot showing the importance of features based on p-values.
    The plot includes both global feature importance and local feature importance for each cluster.
    Global importance is based on all clusters combined, while local importance is specific to each cluster.

    :param p_value_of_features_ranked: DataFrame containing p-values of features, ranked by p-value.
    :type p_value_of_features_ranked: pandas.DataFrame
    :param p_value_of_features_per_cluster: DataFrame containing p-values of features for each cluster.
    :type p_value_of_features_per_cluster: pandas.DataFrame
    :param thr_pvalue: P-value threshold for display. Only features with p-values below this threshold
                    are considered significant. Defaults to 1 (no filtering).
    :type thr_pvalue: float, optional
    :param top_n: Number of top features to display in the plot. If None, all features are included.
                Defaults to None.
    :type top_n: int, optional
    :param num_cols: Number of plots per row in the output figure. Defaults to 4.
    :type num_cols: int, optional
    :param save: Filename to save the plot. If None, the plot will not be saved. Defaults to None.
    :type save: str, optional
    """

    # Determine figure size dynamically based on the number of features
    num_features = len(p_value_of_features_ranked.columns)
    figsize_width = 6.5
    figsize_height = max(figsize_width, int(np.ceil(5 * num_features / 25)))

    num_subplots = 1 + p_value_of_features_per_cluster.shape[1]
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
            "Feature": p_value_of_features_ranked.columns,
            "Importance": log_transform(p_value_of_features_ranked.loc["p_value"].to_list()),
        }
    ).sort_values(by="Importance", ascending=False)

    ax = plt.subplot(num_rows, num_cols, 1)
    sns.barplot(data=importance_global, x="Importance", y="Feature", color="#3470a3")
    ax.axvline(x=log_transform(thr_pvalue), color="red", linestyle="--", label=f"thr p-value = {thr_pvalue}")
    ax.set_title(f"Cluster all")
    ax.legend(bbox_to_anchor=(1, 1), loc=2)

    # Plot local feature importance
    for n, cluster in enumerate(p_value_of_features_per_cluster.columns):
        importance_local = pd.DataFrame(
            {
                "Feature": p_value_of_features_per_cluster.index,
                "Importance": log_transform(p_value_of_features_per_cluster[cluster].to_list()),
            }
        ).sort_values(by="Importance", ascending=False)
        ax = plt.subplot(num_rows, num_cols, n + 2)
        sns.barplot(data=importance_local, x="Importance", y="Feature", color="#3470a3")
        ax.axvline(x=log_transform(thr_pvalue), color="red", linestyle="--")
        ax.set_title(f"Cluster {cluster}")
        # ax.legend(bbox_to_anchor=(1, 1), loc=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save is not None:
        plt.savefig(f"{save}_feature_importance.png", bbox_inches="tight", dpi=300)
    plt.show()


def _plot_heatmap(
    data_clustering_ranked: pd.DataFrame, thr_pvalue: float, top_n: int, model_type: str, save: str
):
    """Plot feature heatmap sorted by clusters, where features are filtered and ranked
    with statistical tests (ANOVA for continuous featres, chi square for categorical features).

    :param data_clustering_ranked: Filtered and ranked data frame incl features, target and cluster numbers.
    :type data_clustering_ranked: pandas.DataFrame
    :param thr_pvalue: P-value threshold used for feature filtering
    :type thr_pvalue: float, optional
    :param model_type: Model type of Random Forest model: classifier or regression.
    :type model_type: str
    :param save: Filename to save plot.
    :type save: str
    """
    data_clustering_ranked = data_clustering_ranked.copy()
    target_values_original = data_clustering_ranked["target"]

    for feature in data_clustering_ranked.columns:
        if pd.api.types.is_string_dtype(data_clustering_ranked[feature]):
            data_clustering_ranked[feature] = data_clustering_ranked[feature].astype("category")
        if isinstance(data_clustering_ranked[feature].dtype, pd.CategoricalDtype) and feature != "cluster":
            data_clustering_ranked[feature] = data_clustering_ranked[feature].cat.codes

    data_scaled = utils.scale_minmax(data_clustering_ranked)
    data_heatmap = pd.DataFrame(columns=data_scaled.columns)
    target_values_scaled = data_scaled["target"]

    one_percent_of_number_of_samples = int(np.ceil(0.01 * len(data_clustering_ranked)))

    for cluster in data_scaled.cluster.unique():
        data_heatmap = pd.concat(
            [data_heatmap, data_scaled[data_scaled.cluster == cluster]],
            ignore_index=True,
        )
        data_heatmap = pd.concat(
            [
                data_heatmap,
                pd.DataFrame(
                    np.nan,
                    index=np.arange(one_percent_of_number_of_samples),
                    # blank lines which are 1% of num samples
                    columns=data_scaled.columns,
                ),
            ],
            ignore_index=True,
        )
    data_heatmap = data_heatmap[:-5]
    data_heatmap.drop("cluster", axis=1, inplace=True)

    n_samples, n_features = data_heatmap.shape
    heatmap_ = np.zeros((n_features, n_samples, 4))
    cmap_features = matplotlib.cm.get_cmap("coolwarm").copy()
    cmap_target = matplotlib.cm.get_cmap("viridis")  # gist_ncar

    for feature in range(n_features):
        for sample in range(n_samples):
            if feature == 0:
                heatmap_[feature, sample, :] = cmap_target(data_heatmap.iloc[sample, feature])
            else:
                heatmap_[feature, sample, :] = cmap_features(data_heatmap.iloc[sample, feature])

    figure_size = max(6.5, int(np.ceil(5 * n_features / 25)))

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))  # Create figure and axes
    img = ax.imshow(heatmap_, interpolation="none", aspect="auto")  # Use the axes for plotting

    plt.suptitle(
        f"Subgroups of instances that follow similar decision paths in the RF model \n Showing {'top ' + str(top_n) if top_n else 'all'} features with p-value < {thr_pvalue}"
    )
    plt.xticks([], [])
    plt.yticks(range(n_features), data_heatmap.columns)

    # remove bounding box
    for spine in ax.spines.values():
        spine.set_visible(False)

    if model_type == "regression":
        norm = matplotlib.colors.Normalize(
            vmin=target_values_original.min(), vmax=target_values_original.max()
        )
        cbar_target = plt.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_target), ax=ax
        )  # Add ax argument
        cbar_target.set_label("target")
    else:
        legend_elements = [
            Patch(
                facecolor=cmap_target(tv_n),
                edgecolor=cmap_target(tv_n),
                label=f"{tv_o}",
            )
            for tv_n, tv_o in zip(target_values_scaled.unique(), target_values_original.unique())
        ]
        ll = ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(0, 0),
            loc="upper left",
            ncol=min(6, len(legend_elements)),
            title="target",
        )

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cbar_features = plt.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_features), ax=ax
    )  # Add ax argument
    cbar_features.set_label("standardized feature values")

    if save is not None:
        plt.savefig("{}_heatmap.png".format(save), bbox_inches="tight", dpi=300)
    plt.show()


def _plot_distributions(
    data_clustering_ranked: pd.DataFrame, thr_pvalue: float, top_n: int, num_cols: int, save: str
):
    """Plot feature boxplots (for continuous features) or barplots (for categorical features) divided by clusters,
    where features are filtered and ranked by p-value of a statistical test (ANOVA for continuous features,
    chi square for categorical features).

    :param data_clustering_ranked: Filtered and ranked data frame incl features, target and cluster numbers.
    :type data_clustering_ranked: pandas.DataFrame
    :param thr_pvalue: P-value threshold used for feature filtering
    :type thr_pvalue: float, optional
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
        f"Distribution of feature values across subgroups - Showing {'top ' + str(top_n) if top_n else 'all'} features with p-value < {thr_pvalue}",
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

    if save is not None:
        plt.savefig(f"{save}_boxplots.png", bbox_inches="tight", dpi=300)
    plt.show()
