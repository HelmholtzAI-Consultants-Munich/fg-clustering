############################################
# Imports
############################################

import os
import uuid

import pandas as pd
import numpy as np

from typing import Optional

from numba import njit, prange

from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

from .utils import check_disk_space

############################################
# Distance Classes
############################################


class DistanceRandomForestProximity:
    def __init__(
        self,
        memory_efficient: Optional[bool] = False,
        dir_distance_matrix: Optional[str] = None,
    ):
        if memory_efficient:
            if dir_distance_matrix is None:
                raise ValueError("You must specify `dir_distance_matrix` when `memory_efficient=True`.")

        self.terminals = None
        self.memory_efficient = memory_efficient
        self.dir_distance_matrix = dir_distance_matrix
        self.precomputed_distance_matrix = None

    def calculate_terminals(self, estimator, X):
        self.terminals = estimator.apply(X).astype(np.int32)

    def calculate_distance_matrix(self, sample_indices=None):

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
    def __init__(self, scale_features):
        self.scale_features = scale_features

    def calculate_distance_cluster_vs_background(self, values_background, values_cluster, is_categorical):
        if is_categorical:
            # Create dummies and make sure that each category gets a column
            dummies_all = pd.get_dummies(values_background, drop_first=True)
            dummies_cluster = pd.get_dummies(values_cluster, drop_first=True)
            dummies_all, dummies_cluster = dummies_all.align(dummies_cluster, join="outer", fill_value=0)

            distances = [
                wasserstein_distance(dummies_all[col], dummies_cluster[col]) for col in dummies_all.columns
            ]
            return np.nanmax(distances)
        else:
            return wasserstein_distance(values_background, values_cluster)


class DistanceJensenShannon:
    def __init__(self, scale_features):
        self.scale_features = scale_features

    def calculate_distance_cluster_vs_background(self, values_background, values_cluster, is_categorical):

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
def _calculate_distances(terminals, n, n_estimators, distance_matrix):

    for i in prange(n):
        for j in range(i + 1, n):  # no prange here due to write conflicts
            proximity = np.sum(terminals[i, :] == terminals[j, :])
            distance = 1.0 - (proximity / n_estimators)
            if distance > 0:
                distance_matrix[i, j] = distance

    return distance_matrix
