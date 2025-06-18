############################################
# Imports
############################################

import os
import uuid

import pandas as pd
import numpy as np

from typing import Optional, Union, List

from numba import njit, prange

from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .utils import check_disk_space

############################################
# Distance Classes
############################################


class DistanceRandomForestProximity:
    """
    Class to compute a proximity-based distance matrix from the terminal nodes of a trained Random Forest model.
    Supports both in-memory and memory-efficient computation via disk-backed memmap arrays.

    :param memory_efficient: Whether to store the distance matrix in a memory-efficient way using a disk-based memmap.
    :type memory_efficient: bool, optional
    :param dir_distance_matrix: Directory path where the distance matrix should be stored when using memory-efficient mode.
    :type dir_distance_matrix: str, optional
    """

    def __init__(
        self,
        memory_efficient: Optional[bool] = False,
        dir_distance_matrix: Optional[str] = None,
    ) -> None:
        """Constructor for the DistanceRandomForestProximity class."""
        if memory_efficient:
            if dir_distance_matrix is None:
                raise ValueError("You must specify `dir_distance_matrix` when `memory_efficient=True`.")

        self.terminals = None
        self.memory_efficient = memory_efficient
        self.dir_distance_matrix = dir_distance_matrix
        self.precomputed_distance_matrix = None

    def calculate_terminals(
        self, estimator: Union[RandomForestClassifier, RandomForestRegressor], X: pd.DataFrame
    ) -> None:
        """
        Calculates and stores the terminal leaf indices of all samples across all trees in the Random Forest.

        :param estimator: A trained Random Forest estimator with the `.apply()` method available (e.g., from sklearn).
        :type estimator: sklearn.ensemble.RandomForestClassifier or RandomForestRegressor
        :param X: Input feature data to pass through the forest to extract terminal nodes.
        :type X: pd.DataFrame
        """
        self.terminals = estimator.apply(X).astype(np.int32)

    def calculate_distance_matrix(self, sample_indices: List = None) -> Union[np.ndarray, np.memmap]:
        """
        Computes the pairwise distance matrix between samples based on terminal leaf similarity.
        Optionally uses a disk-backed memory-mapped array if `memory_efficient=True`.

        :param sample_indices: Indices of the samples to subset the distance matrix calculation. If None, uses all samples.
        :type sample_indices: List or None

        :raises ValueError: If terminal nodes have not been precomputed.
        :raises MemoryError: If insufficient disk space is available to store the memmap distance matrix.

        :return: A symmetric distance matrix (NumPy array or memmap) representing dissimilarities between sample pairs.
        :rtype: np.ndarray or np.memmap
        """
        if self.terminals is not None:
            if sample_indices is not None:
                terminals = self.terminals[sample_indices]
            else:
                terminals = self.terminals
            n, n_estimators = terminals.shape

            if self.memory_efficient:
                buffer_factor = 1.2  # 20% safety buffer
                required_bytes = int(n * n * 4 * buffer_factor)  # float32 = 4 bytes
                if not check_disk_space(self.dir_distance_matrix, required_bytes):
                    raise MemoryError(
                        f"Not enough free space to allocate a {required_bytes / 1e9:.2f} GB memmap distance matrix (with 20% buffer)."
                    )
                self.file_distance_matrix = os.path.join(
                    self.dir_distance_matrix, f"distance_matrix_{uuid.uuid4().hex[:8]}.dat"
                )
                distance_matrix = np.memmap(
                    self.file_distance_matrix, dtype=np.float32, mode="w+", shape=(n, n)
                )
            else:
                distance_matrix = np.zeros((n, n))

            distance_matrix = _calculate_distances(terminals, n, n_estimators, distance_matrix)

            # Ensure symmetry
            distance_matrix += distance_matrix.T
            return distance_matrix
        else:
            raise ValueError(
                "No precomputed terminals available to compute distance matrix! Run `calculate_terminals()` first."
            )


class DistanceWasserstein:
    """
    Class for calculating the Wasserstein distance between the feature distribution in a cluster and the global background.
    Supports both categorical and numerical feature types.

    :param scale_features: Flag indicating whether input features should be scaled.
    :type scale_features: bool
    """

    def __init__(self, scale_features: bool) -> None:
        """Constructor for the DistanceWasserstein class."""
        self.scale_features = scale_features

    def run_scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scales all numeric features in the dataset using standard scaling (without centering the mean).

        :param X: Input DataFrame with mixed feature types.
        :type X: pandas.DataFrame

        :return: Scaled DataFrame with numeric columns transformed.
        :rtype: pandas.DataFrame
        """
        scaler = StandardScaler(with_mean=False)
        numeric_cols = X.select_dtypes(include="number").columns
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        return X

    def calculate_distance_cluster_vs_background(
        self, values_background: pd.Series, values_cluster: pd.Series, is_categorical: bool
    ) -> float:
        """
        Calculates the Wasserstein distance between the feature distribution of a cluster and the global background.
        Handles both categorical and continuous features.

        :param values_background: Feature values from the full dataset (background distribution).
        :type values_background: pandas.Series
        :param values_cluster: Feature values for the current cluster.
        :type values_cluster: pandas.Series
        :param is_categorical: Indicates whether the feature is categorical.
        :type is_categorical: bool

        :return: Wasserstein distance or the maximum distance across dummy variables if categorical.
        :rtype: float
        """
        if is_categorical:
            # Create dummies and make sure that each category gets a column
            dummies_all = pd.get_dummies(values_background, drop_first=False)
            dummies_cluster = pd.get_dummies(values_cluster, drop_first=False)
            dummies_all, dummies_cluster = dummies_all.align(dummies_cluster, join="outer", fill_value=0)

            distances = [
                wasserstein_distance(dummies_all[col], dummies_cluster[col]) for col in dummies_all.columns
            ]
            return np.nanmax(distances)
        else:
            return wasserstein_distance(values_background, values_cluster)


class DistanceJensenShannon:
    """
    Calculates the Jensen-Shannon distance between the feature distribution in a cluster and the global background.
    Supports both numerical and categorical features, using histograms or category frequencies.

    :param scale_features: Flag indicating whether input features should be scaled.
    :type scale_features: bool
    """

    def __init__(self, scale_features: bool) -> None:
        """Constructor for the DistanceJensenShannon class."""
        self.scale_features = scale_features

    def run_scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scales all numeric features in the dataset using standard scaling (without centering the mean).

        :param X: Input DataFrame with mixed feature types.
        :type X: pandas.DataFrame

        :return: Scaled DataFrame with numeric columns transformed.
        :rtype: pandas.DataFrame
        """
        scaler = StandardScaler(with_mean=False)
        numeric_cols = X.select_dtypes(include="number").columns
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        return X

    def calculate_distance_cluster_vs_background(
        self, values_background: pd.Series, values_cluster: pd.Series, is_categorical: bool
    ) -> float:
        """
        Calculates the Jensen-Shannon distance between the feature distribution of a cluster and the background distribution.
        Uses category frequencies for categorical features and histograms for numerical ones.

        :param values_background: Feature values from the full dataset (background distribution).
        :type values_background: pandas.Series
        :param values_cluster: Feature values for the current cluster.
        :type values_cluster: pandas.Series
        :param is_categorical: Indicates whether the feature is categorical.
        :type is_categorical: bool

        :return: Jensen-Shannon distance between the two distributions.
        :rtype: float
        """
        if is_categorical:
            # Extract the values for the two distributions and calculate the distance
            cats = values_background.unique()
            p_ref = values_background.value_counts(normalize=True).reindex(cats, fill_value=0)
            p_cluster = values_cluster.value_counts(normalize=True).reindex(cats, fill_value=0)
            return jensenshannon(p_ref, p_cluster)
        else:
            # Compute number of bins using Freedman-Diaconis rule, enforcing sensible bounds
            range_val = values_background.max() - values_background.min()
            iqr = values_background.quantile(0.75) - values_background.quantile(0.25)
            n_obs = len(values_background)

            if range_val <= 0 or iqr <= 0 or n_obs <= 1:
                bins = 10
            else:
                bin_width = 2 * iqr / (n_obs ** (1 / 3))
                bin_estimate = int(np.ceil(range_val / bin_width))
                bins = max(1, min(bin_estimate, n_obs, 100))

            # Define bin edges and calculate histograms
            edges = np.percentile(values_background, np.linspace(0, 100, bins + 1))
            hist_ref, _ = np.histogram(values_background, bins=edges)
            hist_cluster, _ = np.histogram(values_cluster, bins=edges)

            # Normalize histogram values
            p_ref = (
                hist_ref / np.sum(hist_ref)
                if np.sum(hist_ref) > 0
                else np.ones_like(hist_ref) / len(hist_ref)
            )
            p_cluster = (
                hist_cluster / np.sum(hist_cluster)
                if np.sum(hist_cluster) > 0
                else np.ones_like(hist_cluster) / len(hist_cluster)
            )

            return jensenshannon(p_ref, p_cluster)


############################################
# Numba Functions
############################################


@njit(parallel=True, fastmath=True)
def _calculate_distances(
    terminals: np.ndarray, n: int, n_estimators: int, distance_matrix: Union[np.ndarray, np.memmap]
):
    """
    Computes the upper triangle of a pairwise distance matrix based on Random Forest terminal node similarity.

    The distance between two samples is defined as one minus the fraction of trees in which both samples fall into the same terminal node.
    Only the upper triangle of the matrix is computed to avoid redundant calculations as the resulting matrix is symmetric.

    :param terminals: 2D array of shape (n_samples, n_estimators), where each entry indicates the terminal node index for a sample in a tree.
    :type terminals: np.ndarray
    :param n: Number of samples for which distances should be calculated.
    :type n: int
    :param n_estimators: Number of trees in the Random Forest model.
    :type n_estimators: int
    :param distance_matrix: Pre-allocated 2D array (or memmap) where the resulting distances will be stored.
    :type distance_matrix: np.ndarray or np.memmap

    :return: The updated distance matrix with the upper triangle filled with computed distances.
    :rtype: np.ndarray or np.memmap
    """
    for i in prange(n):
        for j in range(i + 1, n):  # no prange here due to write conflicts
            proximity = np.sum(terminals[i, :] == terminals[j, :])
            distance = 1.0 - (proximity / n_estimators)
            if distance > 0:
                distance_matrix[i, j] = distance

    return distance_matrix
