############################################
# Imports
############################################

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Union, Tuple
from pandas.api.types import is_numeric_dtype

from .distance import DistanceJensenShannon, DistanceWasserstein


############################################
# Feature Importance Class
############################################


class FeatureImportance:
    """
    Calculates the importance of each feature in contributing to the cluster separation based on a
    specified distributional distance metric (e.g., Jensen-Shannon or Wasserstein). This class
    provides both local (per cluster) and global (overall) importance scores by comparing the
    distribution of feature values within clusters against the background (entire dataset).

    :param distance_metric: An instance of DistanceJensenShannon or DistanceWasserstein.
    :type distance_metric: Union[DistanceJensenShannon, DistanceWasserstein]
    """

    def __init__(
        self,
        distance_metric: Union[DistanceJensenShannon, DistanceWasserstein],
    ):
        """Constructor for the FeatureImportance class."""
        self.distance_metric = distance_metric

    def calculate_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cluster_labels: np.ndarray,
        model_type: str,
        verbose: int,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Computes feature importance scores for clustered data using a selected distance metric.

        :param X: Input feature matrix.
        :type X: pandas.DataFrame
        :param y: Target variable.
        :type y: pandas.Series
        :param cluster_labels: Labels from forest-guided clustering.
        :type cluster_labels: numpy.ndarray
        :param model_type: Type of model, either "cla" for classification or "reg" for regression.
        :type model_type: str
        :param verbose: Verbosity level (0 = silent, 1 = progress messages).
        :type verbose: int

        :return: Tuple containing local feature importance scores, global feature importance scores, and the clustering data ordered by the global feature importance.
        :rtype: Tuple[pandas.DataFrame, pandas.Series, pandas.DataFrame]
        """
        self.verbose = verbose

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

    def _calculate_cluster_distance(
        self,
        X: pd.DataFrame,
        cluster_labels: np.ndarray,
    ) -> pd.DataFrame:
        """
        Calculates the distance of each feature between clusters and background.

        :param X: Input feature matrix.
        :type X: pandas.DataFrame
        :param cluster_labels: Labels from forest-guided clustering.
        :type cluster_labels: numpy.ndarray

        :return: A DataFrame of local feature importance scores.
        :rtype: pandas.DataFrame
        """
        clusters_unique = np.unique(cluster_labels)
        distances = pd.DataFrame(index=X.columns, columns=clusters_unique)

        # Optional scaling of numeric values
        if self.distance_metric.scale_features:
            X = self.distance_metric.run_scale_features(X)

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
                or values_background.dtype == "bool"
            ):
                is_categorical = True
            elif is_numeric_dtype(values_background):
                is_categorical = False
            else:
                raise TypeError(f"The type {values_background.dtype} of feature {feature} is not supported.")

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

    def _sort_clusters_by_target(
        self,
        data_clustering_ranked: pd.DataFrame,
        model_type: str,
    ) -> pd.DataFrame:
        """
        Sorts clusters by the mean target value and reorders cluster labels accordingly.

        :param data_clustering_ranked: Combined DataFrame of features, cluster labels, and target values.
        :type data_clustering_ranked: pandas.DataFrame
        :param model_type: Type of model, either "cla" for classification or "reg" for regression.
        :type model_type: str

        :return: Clustering data sorted by target.
        :rtype: pandas.DataFrame
        """
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
