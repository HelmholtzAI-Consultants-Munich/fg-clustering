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
    Quantify how strongly each feature separates clusters from the full data distribution.

    For every feature and cluster, the configured metric compares the distribution of that
    feature **inside the cluster** to the **background** distribution (all rows). Scores are
    column-normalized so the largest distance per cluster is 1. Local scores are aggregated
    (mean across clusters) into a global importance vector, which also drives column ordering
    in the returned clustering table.

    :param distance_metric: Distributional distance implementation (Jensen–Shannon or Wasserstein).
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
        Compute per-feature importance from cluster labels and build a ranked view of the data.

        Joins ``X``, ``y``, optional ``y_pred``, and ``cluster_labels`` into one frame, then
        runs :meth:`_calculate_cluster_distance` for local scores. Global importance is the
        mean of local scores across clusters. The returned clustering frame sorts rows by
        ``cluster``, ``target``, and optional ``predicted_target``, and orders feature columns
        by descending global importance (with those fixed columns first).

        :param X: Feature columns; one row per observation, aligned with ``y`` and ``cluster_labels``.
        :type X: pd.DataFrame
        :param y: Target column, same length as ``X``.
        :type y: pd.Series
        :param y_pred: Optional predictions; if ``None``, ``predicted_target`` is not added.
        :type y_pred: pd.Series | None
        :param cluster_labels: Cluster id per row, same length as ``X``.
        :type cluster_labels: np.ndarray
        :param verbose: Logging verbosity: ``0`` disables the per-feature progress bar and messages when a feature is skipped (e.g. zero variance); any non-zero value enables them.
        :type verbose: int

        :return: ``(feature_importance_local, feature_importance_global, data_clustering_ranked)``. Local: features × clusters. Global: mean importance per feature. Ranked frame: same rows as built above, columns reordered and sorted as described.
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
        For each feature and cluster, measure distributional distance to the full column.

        Numeric vs categorical handling follows dtype checks; constant features (at most one
        distinct value) are skipped and filled with ``NaN``. If ``distance_metric.scale_features``
        is true, numeric columns are scaled via ``distance_metric.run_scale_features`` before
        distances are computed. Each cluster column is divided by its max so the largest score
        in that column is 1 (max zero becomes ``NaN`` to avoid invalid division).

        :param X: Features only (no target or cluster columns).
        :type X: pd.DataFrame
        :param cluster_labels: Label per row of ``X``, used to mask cluster subsets.
        :type cluster_labels: np.ndarray

        :return: Index: feature names; columns: distinct cluster labels; values: normalized distances.
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
