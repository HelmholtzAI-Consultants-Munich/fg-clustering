############################################
# Imports
############################################

import numpy as np
import pandas as pd

from typing import Union

from tqdm import tqdm
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

from sklearn.preprocessing import StandardScaler

from .distance import DistanceJensenShannon, DistanceWasserstein


############################################
# Feature Importance Class
############################################


class FeatureImportance:
    def __init__(self, distance_metric: Union[DistanceJensenShannon, DistanceWasserstein], verbose: int):
        self.distance_metric = distance_metric
        self.verbose = verbose

    def calculate_feature_importance(self, X, y, cluster_labels, model_type):

        data_clustering = pd.concat(
            [X, y.rename("target"), pd.Series(cluster_labels, name="cluster")], axis=1
        )

        feature_importance_local = self._calculate_cluster_distance(X=X, cluster_labels=cluster_labels)

        # Aggregate over all clusters
        feature_importance_global = feature_importance_local.mean(axis=1)

        # Sort features by mean and extract names
        features_ranked = ["cluster", "target"] + feature_importance_global.sort_values(
            ascending=False
        ).index.tolist()

        # Sort and rank clustering dataframe
        data_clustering_ranked = self._sort_clusters_by_target(data_clustering[features_ranked], model_type)
        data_clustering_ranked = data_clustering_ranked.sort_values(by=["cluster", "target"])

        return feature_importance_local, feature_importance_global, data_clustering_ranked

    def _calculate_cluster_distance(self, X, cluster_labels):
        clusters_unique = np.unique(cluster_labels)
        distances = pd.DataFrame(index=X.columns, columns=clusters_unique)

        # Optional scaling of numeric values
        if self.distance_metric.scale_features:
            scaler = StandardScaler(with_mean=False)
            numeric_cols = X.select_dtypes(include="number").columns
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        # Loop through columns
        for feature in tqdm(X.columns, disable=(self.verbose == 0)):
            # Extract all values of column
            values_background = X[feature]

            # Check feature variance
            if values_background.nunique() <= 1:
                if self.verbose:
                    print(f" - Skipping feature with zero variance.")
                distances.loc[feature, :] = np.nan
                continue

            # Check feature type
            if (
                isinstance(values_background.dtype, pd.CategoricalDtype)
                or values_background.dtype == "object"
            ):
                is_categorical = True
            elif is_numeric_dtype(values_background):
                is_categorical = False
            else:
                if self.verbose:
                    print(f"The type {values_background.dtype} is not supported. Skipping feature!")
                distances.loc[feature, :] = np.nan

            # Loop through cluster, extract cluster values
            for cluster in clusters_unique:
                values_cluster = X.loc[cluster_labels == cluster, feature]

                # Calculate distance to background values
                dist = self.distance_metric.calculate_distance_cluster_vs_background(
                    values_background, values_cluster, is_categorical=is_categorical
                )
                distances.loc[feature, cluster] = dist

        # Divide the distance by max for each cluster to get max equal to one
        col_max = distances.max(axis=0).replace(0, np.nan)  # Avoid division by zero
        distances = distances.div(col_max, axis=1)

        distances.columns = [cluster for cluster in clusters_unique]

        return distances

    def _sort_clusters_by_target(self, data_clustering_ranked, model_type):

        # When using a classifier, the target value is label encoded, such that we can sort the clusters by target values
        original_target = data_clustering_ranked["target"].copy()

        if model_type == "cla":
            data_clustering_ranked["target"] = data_clustering_ranked["target"].astype("category").cat.codes

        # Compute mean target values for each cluster and sort by mean values
        cluster_means = data_clustering_ranked.groupby(["cluster"])[["cluster", "target"]].mean()
        cluster_means = cluster_means.sort_values(by="target").index

        # Map the sorted clusters to a new order, replace clusters with the new mapping and ensure the 'cluster' column is a categorical type with ordered levels
        mapping = {cluster: i + 1 for i, cluster in enumerate(cluster_means)}
        data_clustering_ranked["cluster"] = pd.Categorical(
            data_clustering_ranked["cluster"].map(mapping), ordered=True
        )

        # Restore the original target values
        data_clustering_ranked["target"] = original_target

        return data_clustering_ranked
