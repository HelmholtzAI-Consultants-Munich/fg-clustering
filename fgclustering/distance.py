############################################
# Imports
############################################

import os
import gc
import time
import uuid
import numpy as np
import pandas as pd

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
    Compute a proximity-based distance matrix from the terminal nodes of a trained Random Forest model.

    Sample similarity is derived from how often two samples end in the same terminal node
    across trees. Distances are defined as one minus this proximity. The distance matrix
    can be computed fully in memory or stored in a disk-backed memmap array for
    memory-efficient operation.

    :param memory_efficient: Whether to store the distance matrix in a disk-backed memmap array.
    :type memory_efficient: bool
    :param dir_distance_matrix: Directory used to store the memmap distance matrix when ``memory_efficient=True``.
    :type dir_distance_matrix: str | None
    """

    def __init__(
        self,
        memory_efficient: bool = False,
        dir_distance_matrix: str | None = None,
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
        estimator: RandomForestClassifier | RandomForestRegressor,
        X: pd.DataFrame,
    ) -> None:
        """
        Compute and store the terminal node assignments of all samples across all trees.

        The terminal node matrix is obtained by applying the trained Random Forest to ``X``.
        Each row corresponds to a sample and each column to a tree.

        :param estimator: Trained Random Forest estimator.
        :type estimator: RandomForestClassifier | RandomForestRegressor
        :param X: Input feature matrix.
        :type X: pd.DataFrame

        :return: ``None``
        :rtype: None
        """
        self.terminals = estimator.apply(X).astype(np.int32)

    def calculate_distance_matrix(
        self,
        sample_indices: np.ndarray | None,
    ) -> tuple[np.ndarray | np.memmap, str | None]:
        """
        Compute the pairwise distance matrix from Random Forest terminal node assignments.

        The distance between two samples is defined as one minus the fraction of trees in
        which both samples fall into the same terminal node. If ``memory_efficient=True``,
        the distance matrix is created as a disk-backed memmap array after checking that
        sufficient disk space is available.

        :param sample_indices: Indices of the samples for which the distance matrix is computed, or ``None`` to use all samples.
        :type sample_indices: np.ndarray | None

        :raises ValueError: If terminal nodes have not been precomputed.
        :raises MemoryError: If insufficient disk space is available for the memmap distance matrix.

        :return: Tuple containing the distance matrix and the memmap file path, or ``None`` as the path when computed fully in memory.
        :rtype: tuple[np.ndarray | np.memmap, str | None]
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
                distance_matrix = np.zeros((n, n), dtype=np.float32)

            distance_matrix = _calculate_distances(terminals, n, n_estimators, distance_matrix)

            return distance_matrix, file_distance_matrix

    def remove_distance_matrix(
        self,
        distance_matrix: np.ndarray | np.memmap,
        file_distance_matrix: str | None,
    ) -> None:
        """
        Remove a disk-backed distance matrix file and release associated resources.

        If the distance matrix was created as a memmap array, this method attempts to flush
        pending writes, delete the array object, trigger garbage collection, and remove the
        backing file from disk. Repeated removal attempts are made to avoid file-locking
        issues on some systems.

        :param distance_matrix: Distance matrix object to release.
        :type distance_matrix: np.ndarray | np.memmap
        :param file_distance_matrix: Path to the memmap file on disk, or ``None`` if no file was created.
        :type file_distance_matrix: str | None

        :return: ``None``
        :rtype: None
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
    Compute Wasserstein distance between a cluster-specific feature distribution and the background distribution.

    This distance metric supports both numeric and categorical features. Numeric features
    are compared directly, while categorical features are dummy-encoded and compared per
    category, returning the maximum category-wise Wasserstein distance.

    :param scale_features: Whether numeric features should be scaled before distance computation.
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
        Scale numeric feature columns using standard scaling without mean centering.

        Only numeric columns are transformed. Non-numeric columns are left unchanged.

        :param X: Input feature matrix.
        :type X: pd.DataFrame

        :return: Feature matrix with scaled numeric columns.
        :rtype: pd.DataFrame
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
        Compute the Wasserstein distance between the cluster and background distributions of a feature.

        For categorical features, the values are dummy-encoded and the maximum Wasserstein
        distance across categories is returned. For numeric features, the raw feature values
        are compared directly.

        :param values_background: Feature values from the full dataset.
        :type values_background: pd.Series
        :param values_cluster: Feature values from the current cluster.
        :type values_cluster: pd.Series
        :param is_categorical: Whether the feature should be treated as categorical.
        :type is_categorical: bool

        :return: Wasserstein distance between the cluster and background distributions.
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
    Compute Jensen-Shannon distance between a cluster-specific feature distribution and the background distribution.

    This distance metric supports both numeric and categorical features. Categorical
    features are compared using category frequency distributions, while numeric features
    are compared using histogram-based approximations of their distributions.

    :param scale_features: Whether numeric features should be scaled before distance computation.
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
        Scale numeric feature columns using standard scaling without mean centering.

        Only numeric columns are transformed. Non-numeric columns are left unchanged.

        :param X: Input feature matrix.
        :type X: pd.DataFrame

        :return: Feature matrix with scaled numeric columns.
        :rtype: pd.DataFrame
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
        Compute the Jensen-Shannon distance between the cluster and background distributions of a feature.

        For categorical features, the distance is computed from category frequency
        distributions over the categories present in the background data. For numeric
        features, histogram-based distributions are constructed using bin edges derived from
        the background values.

        :param values_background: Feature values from the full dataset.
        :type values_background: pd.Series
        :param values_cluster: Feature values from the current cluster.
        :type values_cluster: pd.Series
        :param is_categorical: Whether the feature should be treated as categorical.
        :type is_categorical: bool

        :return: Jensen-Shannon distance between the cluster and background distributions.
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
            p_ref = hist_ref / np.sum(hist_ref)
            p_cluster = hist_cluster / np.sum(hist_cluster)

            return jensenshannon(p_ref, p_cluster)


############################################
# Numba Functions
############################################


@njit(parallel=True)
def _calculate_distances(
    terminals: np.ndarray,
    n: int,
    n_estimators: int,
    distance_matrix: np.ndarray | np.memmap,
) -> np.ndarray | np.memmap:
    """
    Compute the symmetric pairwise distance matrix from Random Forest terminal node assignments.

    The distance between two samples is defined as one minus the fraction of trees in
    which both samples fall into the same terminal node. The upper triangle is computed
    first, and the lower triangle is filled in a second pass to ensure symmetry.

    :param terminals: Array of terminal node assignments with shape ``(n_samples, n_estimators)``.
    :type terminals: np.ndarray
    :param n: Number of samples.
    :type n: int
    :param n_estimators: Number of trees in the Random Forest.
    :type n_estimators: int
    :param distance_matrix: Pre-allocated array in which the pairwise distances are stored.
    :type distance_matrix: np.ndarray | np.memmap

    :return: Symmetric pairwise distance matrix.
    :rtype: np.ndarray | np.memmap
    """
    for i in prange(n):
        for j in range(i + 1, n):
            # use explicit loop for proximity to avoid temporary array allocation and minimize memory traffic
            proximity = 0
            for t in range(n_estimators):
                if terminals[i, t] == terminals[j, t]:
                    proximity += 1

            distance = 1.0 - (proximity / n_estimators)
            distance_matrix[i, j] = distance

    # Fill lower triangle in a separate pass to ensure symmetry without race conditions (if done inside prange loop) and without allocating a full transpose.
    for i in range(n):
        for j in range(i + 1, n):
            distance_matrix[j, i] = distance_matrix[i, j]

    return distance_matrix
