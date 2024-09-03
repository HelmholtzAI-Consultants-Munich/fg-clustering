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


def _plot_global_feature_importance(p_value_of_features_ranked, top_n, save):
    """Plot global feature importance based on p-values given as input.

    :param p_value_of_features: dictionary where keys are names of features and values are p-values of these features
    :type p_value_of_features: dict
    :param save: Filename to save plot.
    :type save: str
    """
    # Transform the DataFrame and calculate feature importance
    importance = pd.melt(p_value_of_features_ranked, var_name="Feature", value_name="p_value")
    importance["Importance"] = 1 - importance["p_value"]
    importance.sort_values(by="Importance", ascending=False, inplace=True)

    # Determine figure size dynamically based on the number of features
    n_features = len(importance)
    figure_height = max(6.5, int(np.ceil(5 * n_features / 25)))

    # Plotting
    plt.figure(figsize=(6.5, figure_height))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=importance, x="Importance", y="Feature", color="#3470a3")

    plt.title(f"Global Feature Importance for {'top ' + str(top_n) if top_n else 'all'} features")
    plt.tight_layout()

    # Save plot if a filename is provided
    if save:
        plt.savefig(f"{save}_global_feature_importance.png", bbox_inches="tight", dpi=300)

    plt.show()


def _plot_local_feature_importance(p_value_of_features_per_cluster, thr_pvalue, top_n, num_cols, save):
    """Plot local feature importance to show the importance of each feature for each cluster.

    :param p_value_of_features_per_cluster: p-value matrix of all features per cluster.
    :type p_value_of_features_per_cluster: pandas.DataFrame
    :param thr_pvalue: P-value threshold used for feature filtering
    :type thr_pvalue: float, optional
    :param num_cols: Number of plots in one row.
    :type num_cols: int
    :param save: Filename to save plot.
    :type save: str
    """

    importance = 1 - p_value_of_features_per_cluster

    # Reshape and sort the data
    X_barplot = (
        importance.melt(ignore_index=False, var_name="Cluster", value_name="Importance")
        .rename_axis("Feature")
        .reset_index(level=0, inplace=False)
        .sort_values("Importance", ascending=False)
    )

    # Determine figure size based on the number of features and clusters
    n_features = len(X_barplot["Feature"].unique())
    figure_height = max(6.5, int(np.ceil(5 * n_features / 25)))
    num_cols = min(num_cols, len(importance.columns))

    # Set up the Seaborn theme and create the FacetGrid
    sns.set_theme(style="whitegrid")
    g = sns.FacetGrid(X_barplot, col="Cluster", sharey=False, col_wrap=num_cols, height=figure_height)
    g.map(sns.barplot, "Importance", "Feature", order=None, color="#3470a3")

    # Label the axes and set the plot titles
    g.set_titles(col_template="Cluster {col_name}")
    plt.suptitle(
        f"Local Feature Importance - Showing {'top ' + str(top_n) if top_n else 'all'} features with p-value < {thr_pvalue}",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the plot if a save path is provided
    if save:
        plt.savefig(f"{save}_local_feature_importance.png", bbox_inches="tight", dpi=300)

    plt.show()


def _plot_heatmap(data_clustering_ranked, thr_pvalue, top_n, model_type, save):
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


def _plot_distributions(data_clustering_ranked, thr_pvalue, top_n, num_cols, save):
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
