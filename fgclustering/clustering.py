############################################
# Imports
############################################

import os
import kmedoids
import numpy as np

from numba import njit, prange

from sklearn.utils import check_random_state

from .distance import DistanceRandomForestProximity
from .utils import check_sub_sample_size

############################################
# Clustering Classes
############################################


class ClusteringKMedoids:
    def __init__(
        self,
        method: str = "fasterpam",
        init: str = "random",
        max_iter: int = 100,
        random_state: int = 42,
    ):
        self.method = method
        self.init = init
        self.metric = "precomputed"
        self.max_iter = max_iter
        self.random_state = random_state

    def run_clustering(
        self,
        k: int,
        distance_metric: DistanceRandomForestProximity,
        sample_indices: np.ndarray,
    ):

        distance_matrix = distance_metric.calculate_distance_matrix(sample_indices=sample_indices)

        cluster_labels = (
            kmedoids.KMedoids(
                n_clusters=k,
                method=self.method,
                init=self.init,
                metric=self.metric,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            .fit(distance_matrix)
            .labels_
        )

        if distance_metric.memory_efficient:
            os.remove(distance_metric.file_distance_matrix)

        return cluster_labels + 1


class ClusteringClara:
    def __init__(
        self,
        sub_sample_size: int = None,
        sampling_iter: int = None,
        method: str = "fasterpam",
        init: str = "random",
        max_iter: int = 100,
        random_state: int = 42,
    ):
        self.sub_sample_size = sub_sample_size
        self.sampling_iter = sampling_iter
        self.method = method
        self.init = init
        self.metric = "precomputed"
        self.max_iter = max_iter
        self.random_state = random_state
        self.random_state_ = check_random_state(random_state)

    def run_clustering(
        self,
        k: int,
        distance_metric: DistanceRandomForestProximity,
        sample_indices: np.ndarray,
    ):

        # the input sample indices, aren't neccesarily indices from 1 to n
        # but can be subsampled already in the JI bootstrap, hence we need to
        # define a new index mapping for the potentially subsampled input indices
        n_samples = len(sample_indices)
        sample_indices_mapping = np.arange(n_samples)

        # initialize score (the lower the better)
        best_score = np.inf

        # check if the input sub sample size is valid
        sub_sample_size = check_sub_sample_size(sub_sample_size=self.sub_sample_size, n_samples=n_samples)
        if self.sampling_iter is None:
            self.sampling_iter = max(5, int(np.log2(n_samples)))

        # iterate n times over the input dataset
        for i in range(self.sampling_iter):
            sub_sample_indices = self.random_state_.choice(
                sample_indices_mapping, size=sub_sample_size, replace=False
            )
            sub_sample_indices = np.sort(sub_sample_indices)

            # distance matrix for subsample but input original indices
            sub_sample_distance_matrix = distance_metric.calculate_distance_matrix(
                sample_indices=sample_indices[sub_sample_indices]
            )

            # call k medoids
            kmedoids_subsample = kmedoids.KMedoids(
                n_clusters=k,
                method=self.method,
                init=self.init,
                metric=self.metric,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            kmedoids_subsample.fit(sub_sample_distance_matrix)

            # remove distance matrix if finished
            if distance_metric.memory_efficient:
                os.remove(distance_metric.file_distance_matrix)

            # retrieve the calculated medoids and use to calculate inertia score
            sub_sample_medoids_idxs = sub_sample_indices[kmedoids_subsample.medoid_indices_]
            sub_sample_score = _calculate_inertia(
                distance_metric.terminals, sample_indices, sample_indices[sub_sample_medoids_idxs]
            )

            # update if score is better
            if sub_sample_score < best_score:
                best_score = sub_sample_score
                best_medoids_idxs = sub_sample_medoids_idxs

        # assign labels to the rest of the data when best medoids are found
        cluster_labels = _asign_labels(
            distance_metric.terminals, sample_indices, sample_indices[best_medoids_idxs]
        )

        return cluster_labels


############################################
# Numba Functions
############################################


@njit(parallel=True, fastmath=True)
def _calculate_inertia(terminals, sample_idx, medoids_idx):

    n_estimators = terminals.shape[1]
    n_samples = len(sample_idx)
    inertia = 0

    for i in prange(n_samples):
        sample = sample_idx[i]
        distance_sample_min = np.inf

        for medoid in medoids_idx:
            proximity = np.sum(terminals[sample, :] == terminals[medoid, :])
            distance_sample = 1.0 - (proximity / n_estimators)
            if distance_sample < distance_sample_min:
                distance_sample_min = distance_sample

        inertia += distance_sample_min

    return inertia


@njit(parallel=True, fastmath=True)
def _asign_labels(terminals, sample_idx, medoids_idx):

    n_estimators = terminals.shape[1]
    n_samples = len(sample_idx)
    n_medoids = len(medoids_idx)
    cluster_labels = np.zeros(n_samples, dtype=np.int16)

    for i in prange(n_samples):
        sample = sample_idx[i]
        cluster_label_sample = np.zeros(n_medoids, dtype=np.float32)

        for j in range(n_medoids):
            medoid = medoids_idx[j]
            proximity = np.sum(terminals[sample, :] == terminals[medoid, :])
            cluster_label_sample[j] = 1.0 - (proximity / n_estimators)

        cluster_labels[i] = np.argmin(cluster_label_sample)

    return cluster_labels
