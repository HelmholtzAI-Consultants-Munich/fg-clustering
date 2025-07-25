############################################################
##### Imports
############################################################

import os
import umap
import kmedoids
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from collections import Counter

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


############################################################
##### Utility Functions
############################################################

palette_dict = {
    "Condition": ["lightgrey", "grey"],
    "Tissue": ["darkseagreen", "darkgreen"],
    "GSE": sns.color_palette("tab20") + sns.color_palette("tab10"),
    "Disease": sns.color_palette("tab20b") + sns.color_palette("tab20c"),
    "Set": ["cornflowerblue", "darkblue"],
}

plt.rcParams.update({"font.size": 12})

dir_output = "results"


def plot_pie_charts(dataset, columns_to_plot=["Condition", "Tissue", "GSE", "Disease"], name=""):

    n_cols = len(columns_to_plot)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))

    for i, col in enumerate(columns_to_plot):
        sorted_labels = sorted(dataset[col].dropna().unique())  # sort for consistency
        counts = {label: (dataset[col] == label).sum() for label in sorted_labels}
        values = list(counts.values())
        labels = list(counts.keys())

        color_list = palette_dict[col][: len(labels)]

        if len(labels) <= 10:
            wedges, texts, autotext = axes[i].pie(
                values,
                labels=[None] * len(labels),
                autopct="%1.1f%%",
                startangle=140,
                wedgeprops=dict(edgecolor="white"),
                colors=color_list,
            )
        else:
            wedges, texts = axes[i].pie(
                values,
                labels=[None] * len(labels),
                autopct=None,
                startangle=140,
                wedgeprops=dict(edgecolor="white"),
                colors=color_list,
            )

        axes[i].set_title(f"Distribution of {col}")
        axes[i].axis("equal")
        axes[i].legend(
            handles=wedges,
            labels=labels,
            title=col,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2 if len(labels) > 10 else 1,
            fontsize="small",
        )

    # plt.tight_layout()
    plt.savefig(os.path.join(dir_output, f"{name}_pie_chart.pdf"), format="pdf", bbox_inches="tight")
    plt.show()


def plot_pca(dataset, columns_to_plot=["Condition", "Tissue", "GSE", "Disease"], name=""):
    metadata_cols = ["sample_id", "Dataset", "GSE", "Condition", "Disease", "Tissue"]
    expression = dataset.drop(columns=metadata_cols, errors="ignore")
    metadata = dataset[columns_to_plot]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(expression)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df = pd.concat([pca_df, metadata.reset_index(drop=True)], axis=1)

    n_cols = len(columns_to_plot)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), squeeze=False)

    for i, col in enumerate(columns_to_plot):
        unique_classes = sorted(dataset[col].unique())  # sort for consistency
        palette = {cls: palette_dict[col][j] for j, cls in enumerate(unique_classes)}

        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue=col,
            alpha=0.7,
            edgecolor=None,
            palette=palette,
            ax=axes[0, i],
        )
        axes[0, i].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        axes[0, i].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        axes[0, i].set_title(f"PCA colored by {col}")
        axes[0, i].get_legend().remove()
        # ax.legend(markerscale=1.5, fontsize="x-mall", loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(dir_output, f"{name}_pca.pdf"), format="pdf", bbox_inches="tight")
    plt.show()


def plot_umap(dataset, columns_to_plot=["Condition", "Tissue", "GSE", "Disease"], name=""):
    metadata_cols = ["sample_id", "Dataset", "GSE", "Condition", "Disease", "Tissue"]
    gene_expression = dataset.drop(columns=metadata_cols)

    # Normalize the gene expression data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(gene_expression)

    # Run UMAP dimensionality reduction
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    n_cols = len(columns_to_plot)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))

    for i, col in enumerate(columns_to_plot):
        unique_classes = sorted(dataset[col].unique())  # sort for consistency
        palette = {cls: palette_dict[col][j] for j, cls in enumerate(unique_classes)}

        for cls in unique_classes:
            indices = dataset[col] == cls
            axes[i].scatter(
                embedding[indices, 0],
                embedding[indices, 1],
                label=str(cls),
                color=palette[cls],
                alpha=0.7,
                s=20,
            )

        axes[i].set_title(f"UMAP colored by {col}")
        axes[i].set_xlabel("UMAP-1")
        axes[i].set_ylabel("UMAP-2")
        # axes[i].legend(markerscale=1.5, fontsize="x-small", loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(dir_output, f"{name}_umap.pdf"), format="pdf", bbox_inches="tight")
    plt.show()


def hierarchical_clustering_heatmap(dataset, name=""):
    metadata_cols = ["sample_id", "Dataset", "GSE", "Condition", "Disease", "Tissue"]
    gene_expression = dataset.drop(columns=metadata_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(gene_expression)
    X_scaled_df = pd.DataFrame(X_scaled, index=dataset.index, columns=gene_expression.columns)

    condition_values = dataset["Condition"].astype(str)
    disease_values = dataset["Disease"].astype(str)
    study_values = dataset["GSE"].astype(str)

    sorted_condition = sorted(condition_values.unique())
    sorted_disease = sorted(disease_values.unique())
    sorted_study = sorted(study_values.unique())

    condition_palette = palette_dict["Condition"][: len(sorted_condition)]
    disease_palette = palette_dict["Disease"][: len(sorted_disease)]
    study_palette = palette_dict["GSE"][: len(sorted_study)]

    condition_lut = dict(zip(sorted_condition, condition_palette))
    disease_lut = dict(zip(sorted_disease, disease_palette))
    study_lut = dict(zip(sorted_study, study_palette))

    row_colors = pd.DataFrame(
        {
            "Condition": condition_values.map(condition_lut),
            "Disease": disease_values.map(disease_lut),
            "GSE": study_values.map(study_lut),
        },
        index=dataset.index,
    )

    g = sns.clustermap(
        X_scaled_df,
        method="average",
        metric="euclidean",
        row_colors=row_colors,
        figsize=(16, 10),
        row_cluster=False,
        col_cluster=False,
        cmap="coolwarm",
        xticklabels=False,
        yticklabels=False,
    )

    # condition_patches = [Patch(color=condition_lut[c], label=c) for c in sorted_condition]
    # disease_patches = [Patch(color=disease_lut[d], label=d) for d in sorted_disease]

    # g.ax_heatmap.legend(
    #    handles=condition_patches + disease_patches,
    #    bbox_to_anchor=(1.05, 1),
    #    loc="upper left",
    #    fontsize="small",
    #    title="Annotations",
    # )

    plt.title("Hierarchical clustering on full dataset", y=1.05)
    plt.savefig(
        os.path.join(dir_output, f"{name}_hierarchical_clustering.png"),
        format="png",
        bbox_inches="tight",
        dpi=800,
    )
    plt.show()


def kmedoids_clustering_heatmap(dataset, k=2, name=""):
    metadata_cols = ["sample_id", "Dataset", "GSE", "Condition", "Disease", "Tissue"]
    gene_expression = dataset.drop(columns=metadata_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(gene_expression)
    X_scaled_df = pd.DataFrame(X_scaled, index=dataset.index, columns=gene_expression.columns)

    clustering = kmedoids.KMedoids(n_clusters=k, random_state=42, method="pam", metric="euclidean")
    cluster_labels = clustering.fit_predict(X_scaled_df.to_numpy())
    dataset["cluster"] = cluster_labels

    sorted_idx = dataset.sort_values("cluster").index
    X_sorted = X_scaled_df.loc[sorted_idx]

    condition_values = dataset.loc[sorted_idx, "Condition"].astype(str)
    disease_values = dataset.loc[sorted_idx, "Disease"].astype(str)
    study_values = dataset.loc[sorted_idx, "GSE"].astype(str)

    sorted_condition = sorted(condition_values.unique())
    sorted_disease = sorted(disease_values.unique())
    sorted_study = sorted(study_values.unique())

    condition_palette = palette_dict["Condition"][: len(sorted_condition)]
    disease_palette = palette_dict["Disease"][: len(sorted_disease)]
    study_palette = palette_dict["GSE"][: len(sorted_study)]

    condition_lut = dict(zip(sorted_condition, condition_palette))
    disease_lut = dict(zip(sorted_disease, disease_palette))
    study_lut = dict(zip(sorted_study, study_palette))

    row_colors = pd.DataFrame(
        {
            "Condition": condition_values.map(condition_lut),
            "Disease": disease_values.map(disease_lut),
            "GSE": study_values.map(study_lut),
        },
        index=sorted_idx,
    )

    g = sns.clustermap(
        X_sorted,
        row_colors=row_colors,
        col_cluster=False,
        row_cluster=False,
        cmap="coolwarm",
        figsize=(16, 10),
        xticklabels=False,
        yticklabels=False,
    )

    condition_patches = [Patch(color=condition_lut[c], label=c) for c in sorted_condition]
    disease_patches = [Patch(color=disease_lut[d], label=d) for d in sorted_disease]
    g.ax_heatmap.legend(
        handles=condition_patches + disease_patches,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize="small",
        title="Annotations",
    )

    cluster_sorted = dataset.loc[sorted_idx, "cluster"].values
    cluster_change_indices = [
        i for i in range(1, len(cluster_sorted)) if cluster_sorted[i] != cluster_sorted[i - 1]
    ]

    row_color_axes = g.ax_row_colors
    if not isinstance(row_color_axes, (list, np.ndarray)):
        row_color_axes = [row_color_axes]

    for i in cluster_change_indices:
        g.ax_heatmap.axhline(i, color="white", linewidth=2)
        for ax in row_color_axes:
            ax.axhline(i, color="white", linewidth=2)

    plt.title(f"K-Medoids clustering (k={k}) on full dataset", y=1.05)
    plt.savefig(
        os.path.join(dir_output, f"{name}_kmedoids_clustering.png"),
        format="png",
        bbox_inches="tight",
        dpi=800,
    )
    plt.show()

    return dataset


def plot_stacked_bar_chart(feature_importance, columns_to_plot=["Tissue", "GSE", "Disease"], name=""):

    data_clustering = feature_importance.data_clustering.copy()

    fig, axes = plt.subplots(1, len(columns_to_plot), figsize=(28, 6))
    for i, feature in enumerate(columns_to_plot):
        sorted_categories = sorted(data_clustering[feature].dropna().unique())
        data_clustering[feature] = pd.Categorical(
            data_clustering[feature], categories=sorted_categories, ordered=True
        )

        counts = data_clustering.groupby(["cluster", feature], observed=True).size().unstack(fill_value=0)
        counts = counts[sorted_categories]  # Ensure consistent order
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100

        num_categories = len(sorted_categories)
        colors = palette_dict[feature][:num_categories]
        cmap = ListedColormap(colors)

        percentages.plot(kind="bar", stacked=True, ax=axes[i], width=0.6, colormap=cmap)

        axes[i].set_ylabel("Percentage")
        axes[i].set_title(f"Stacked Bar Chart by Cluster for {feature}")
        axes[i].get_legend().remove()
        axes[i].legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(dir_output, f"{name}_stacked_bar_chart.pdf"), format="pdf", bbox_inches="tight")
    plt.show()
