############################################
# Imports
############################################

import os
import gc
import time
import uuid
import numpy as np
import pandas as pd

from typing import Optional, Union
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

    :param memory_efficient: Whether to store the distance matrix in a memory-efficient way using a disk-based memmap. Default: False.
    :type memory_efficient: Optional[bool]
    :param dir_distance_matrix: Directory path where the distance matrix should be stored when using memory-efficient mode. Default: None.
    :type dir_distance_matrix: Optional[str]
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
        self,
        estimator: Union[RandomForestClassifier, RandomForestRegressor],
        X: pd.DataFrame,
    ) -> None:
        """
        Calculates and stores the terminal nodes of all samples across all trees in the Random Forest.

        :param estimator: A trained Random Forest estimator from sklearn.
        :type estimator: Union[sklearn.ensemble.RandomForestClassifier, RandomForestRegressor]
        :param X: Input feature matrix.
        :type X: pandas.DataFrame
        """
        self.terminals = estimator.apply(X).astype(np.int32)

    def calculate_distance_matrix(
        self,
        sample_indices: Union[np.ndarray, None],
    ) -> Union[np.ndarray, np.memmap]:
        """
        Computes the pairwise distance matrix between samples based on terminal node similarity
        derived from Random Forest models, where distance is defined as one minus the fraction
        of trees in which a pair of samples fall into the same terminal node.
        Optionally uses a disk-backed memmap array if `memory_efficient=True`.

        :param sample_indices: Indices of samples for which distance is calculated. If None, uses all samples.
        :type sample_indices: Union[np.ndarray, None]

        :raises ValueError: If terminal nodes have not been precomputed.
        :raises MemoryError: If insufficient disk space is available to store the memmap distance matrix.

        :return: A symmetric distance matrix with pairwise distances between samples.
        :rtype: Union[numpy.ndarray, numpy.memmap]
        """
        if self.terminals is None:
            raise ValueError(
                "No precomputed terminals available to compute distance matrix! Run `calculate_terminals()` first."
            )
        else:
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
                file_distance_matrix = os.path.join(
                    self.dir_distance_matrix, f"distance_matrix_{uuid.uuid4().hex[:8]}.dat"
                )
                distance_matrix = np.memmap(file_distance_matrix, dtype=np.float32, mode="w+", shape=(n, n))
            else:
                file_distance_matrix = None
                distance_matrix = np.zeros((n, n))

            distance_matrix = _calculate_distances(terminals, n, n_estimators, distance_matrix)

            # Ensure symmetry
            distance_matrix += distance_matrix.T
            return distance_matrix, file_distance_matrix

    def remove_distance_matrix(
        self,
        distance_matrix: Union[np.ndarray, np.memmap],
        file_distance_matrix: str,
    ) -> None:
        """
        Removes the disk-backed distance matrix file if memory-efficient mode is enabled
        and a distance matrix file was created. Ensures proper file handle release to avoid
        file locking issues, especially on Windows.

        :param distance_matrix: The distance matrix object to delete.
        :type distance_matrix: Union[numpy.ndarray, numpy.memmap]
        :param file_distance_matrix: Full path to the memmap file on disk.
        :type file_distance_matrix: str
        """
        try:
            distance_matrix.flush()
        except Exception:
            pass  # Might not always be necessary, but safe to attempt

        del distance_matrix
        gc.collect()

        if file_distance_matrix is not None and os.path.exists(file_distance_matrix):
            for _ in range(3):
                try:
                    os.remove(file_distance_matrix)
                    break
                except PermissionError:
                    time.sleep(0.5)  # Give OS time to release the file
                    gc.collect()


class DistanceWasserstein:
    """
    Class for calculating the Wasserstein distance between the feature distribution in a cluster and the
    global background distribution of that feature. Supports both categorical and numerical feature types.

    :param scale_features: Flag indicating whether input features should be scaled.
    :type scale_features: bool
    """

    def __init__(
        self,
        scale_features: bool,
    ) -> None:
        """Constructor for the DistanceWasserstein class."""
        self.scale_features = scale_features

    def run_scale_features(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Scales all numeric features in the dataset using standard scaling (without centering the mean).

        :param X: Input feature matrix.
        :type X: pandas.DataFrame

        :return: Input feature matrix with numeric columns transformed.
        :rtype: pandas.DataFrame
        """
        scaler = StandardScaler(with_mean=False)
        numeric_cols = X.select_dtypes(include="number").columns
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        return X

    def calculate_distance_cluster_vs_background(
        self,
        values_background: pd.Series,
        values_cluster: pd.Series,
        is_categorical: bool,
    ) -> float:
        """
        Calculates the Wasserstein distance between the feature distribution in a cluster and the
        global background distribution of that feature. Uses dummy-encoded binary vectors for
        categorical features (returning max distance) and raw values for numerical ones.

        :param values_background: Feature values from the full dataset (background distribution).
        :type values_background: pandas.Series
        :param values_cluster: Feature values for the current cluster.
        :type values_cluster: pandas.Series
        :param is_categorical: Indicates whether the feature is categorical.
        :type is_categorical: bool

        :return: Wasserstein distance between the two distributions.
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
    Class for calculating the Jensen-Shannon distance between the feature distribution in a cluster and the
    global background distribution of that feature. Supports both categorical and numerical feature types.

    :param scale_features: Flag indicating whether input features should be scaled.
    :type scale_features: bool
    """

    def __init__(
        self,
        scale_features: bool,
    ) -> None:
        """Constructor for the DistanceJensenShannon class."""
        self.scale_features = scale_features

    def run_scale_features(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Scales all numeric features in the dataset using standard scaling (without centering the mean).

        :param X: Input feature matrix.
        :type X: pandas.DataFrame

        :return: Input feature matrix with numeric columns transformed.
        :rtype: pandas.DataFrame
        """
        scaler = StandardScaler(with_mean=False)
        numeric_cols = X.select_dtypes(include="number").columns
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        return X

    def calculate_distance_cluster_vs_background(
        self,
        values_background: pd.Series,
        values_cluster: pd.Series,
        is_categorical: bool,
    ) -> float:
        """
        Calculates the Jensen-Shannon distance between the feature distribution in a cluster and the
        global background distribution of that feature. Uses category frequencies for categorical
        features and histograms for numerical ones.

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
    terminals: np.ndarray,
    n: int,
    n_estimators: int,
    distance_matrix: Union[np.ndarray, np.memmap],
) -> Union[np.ndarray, np.memmap]:
    """
    Computes the upper triangle of a pairwise distance matrix based on Random Forest terminal node similarity.

    The distance between two samples is defined as one minus the fraction of trees in which both samples
    fall into the same terminal node. Only the upper triangle of the matrix is computed to avoid redundant
    calculations as the resulting matrix is symmetric.

    :param terminals: 2D array of shape (n_samples, n_estimators), where each entry indicates the terminal node index for a sample in a tree.
    :type terminals: numpy.ndarray
    :param n: Number of samples.
    :type n: int
    :param n_estimators: Number of trees in the Random Forest model.
    :type n_estimators: int
    :param distance_matrix: Pre-allocated 2D array where the resulting distances will be stored.
    :type distance_matrix: Union[numpy.ndarray, numpy.memmap]

    :return: The updated distance matrix with the upper triangle filled with computed distances.
    :rtype: Union[numpy.ndarray, numpy.memmap]
    """
    for i in prange(n):
        for j in range(i + 1, n):  # no prange here due to write conflicts
            proximity = np.sum(terminals[i, :] == terminals[j, :])
            distance = 1.0 - (proximity / n_estimators)
            if distance > 0:
                distance_matrix[i, j] = distance

    return distance_matrix
