############################################
# Imports
############################################

import numpy as np
import pandas as pd

from tqdm import tqdm
from pandas.api.types import is_numeric_dtype, is_string_dtype

from .distance import DistanceJensenShannon, DistanceWasserstein


############################################
# Feature Importance Class
############################################


class FeatureImportance:
    """
    Compute feature importance from cluster-vs-background distribution shifts.

    For each feature and cluster, the configured distance metric compares the feature
    distribution within the cluster to the feature distribution across the full dataset.
    These cluster-specific scores form the local feature importance matrix.

    Local scores are normalized within each cluster so that the largest feature distance is
    ``1``. Global feature importance is computed as the mean local importance across
    clusters and is used to rank feature columns in the returned clustering table.

    :param distance_metric: Distance metric used to compare cluster and background feature
        distributions.
    :type distance_metric: DistanceJensenShannon | DistanceWasserstein
    """

    def __init__(
        self,
        distance_metric: DistanceJensenShannon | DistanceWasserstein,
    ):
        """Constructor for the FeatureImportance class."""
        self.distance_metric = distance_metric

    def calculate_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        y_pred: pd.Series | None,
        cluster_labels: np.ndarray,
        verbose: int,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Calculate local and global feature importance.

        The feature matrix, target values, optional predictions, and cluster labels are combined
        into a clustering table. Local feature importance is computed as normalized
        cluster-vs-background distances for each feature and cluster. Global feature importance
        is the mean local importance across clusters.

        The returned clustering table is sorted by ``cluster``, ``target``, and, if present,
        ``predicted_target``. Feature columns are ordered by descending global importance.

        :param X: Feature matrix with one row per sample.
        :type X: pd.DataFrame
        :param y: Target values aligned with ``X``.
        :type y: pd.Series
        :param y_pred: Optional predicted target values aligned with ``X``.
        :type y_pred: pd.Series | None
        :param cluster_labels: Cluster labels aligned with ``X``.
        :type cluster_labels: np.ndarray
        :param verbose: Verbosity level for progress output.
        :type verbose: int

        :return: Tuple containing local feature importance, global feature importance, and the
            ranked clustering table.
        :rtype: tuple[pd.DataFrame, pd.Series, pd.DataFrame]
        """
        self.verbose = verbose

        data_clustering = pd.concat(
            [X]
            + [y.rename("target")]
            + ([y_pred.rename("predicted_target")] if y_pred is not None else [])
            + [pd.Series(cluster_labels, name="cluster")],
            axis=1,
        )

        feature_importance_local = self._calculate_cluster_distance(X=X, cluster_labels=cluster_labels)

        # Aggregate over all clusters
        feature_importance_global = feature_importance_local.mean(axis=1)

        # Sort features by mean and extract names
        fixed_cols = ["cluster", "target"] + (["predicted_target"] if y_pred is not None else [])
        features_ranked = fixed_cols + feature_importance_global.sort_values(ascending=False).index.tolist()

        # Sort and rank clustering dataframe
        data_clustering_ranked = data_clustering[features_ranked]
        data_clustering_ranked = data_clustering_ranked.sort_values(by=fixed_cols)

        return feature_importance_local, feature_importance_global, data_clustering_ranked

    def _calculate_cluster_distance(
        self,
        X: pd.DataFrame,
        cluster_labels: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compute normalized feature distances for each cluster.

        Each feature distribution within a cluster is compared against the corresponding
        background distribution across all samples using the configured distance metric.
        Categorical and numeric features are detected from their dtype. Features with zero
        variance are skipped and filled with ``NaN``.

        If the distance metric enables feature scaling, numeric columns are scaled before
        distances are computed. Distances are normalized independently for each cluster by the
        maximum feature distance in that cluster.

        :param X: Feature matrix without target, prediction, or cluster columns.
        :type X: pd.DataFrame
        :param cluster_labels: Cluster labels aligned with ``X``.
        :type cluster_labels: np.ndarray

        :raises TypeError: If a feature dtype is not supported.

        :return: Normalized feature-importance matrix with features as rows and clusters as
            columns.
        :rtype: pd.DataFrame
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
                or is_string_dtype(values_background)
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
