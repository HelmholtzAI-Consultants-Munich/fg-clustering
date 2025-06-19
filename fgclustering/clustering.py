############################################
# Imports
############################################

import os
import kmedoids
import numpy as np

from typing import Optional

from numba import njit, prange

from sklearn.utils import check_random_state

from .distance import DistanceRandomForestProximity
from .utils import check_sub_sample_size

############################################
# Clustering Classes
############################################


class ClusteringKMedoids:
    """
    Perform clustering using the K-Medoids algorithm based on a distance matrix
    computed from Random Forest terminal node proximity.

    The class assumes that terminal nodes are precomputed and stored in the provided
    DistanceRandomForestProximity object. The actual distance matrix is computed on the fly
    using these terminals, which helps reduce memory usage, especially for large datasets.

    :param method: Method used in K-Medoids clustering (e.g., 'pam' or 'fasterpam').
    :type method: str
    :param init: Initialization strategy for K-Medoids (e.g., 'random' or 'build').
    :type init: str
    :param max_iter: Maximum number of iterations for the K-Medoids algorithm.
    :type max_iter: int
    :param random_state: Random seed for reproducibility.
    :type random_state: int
    """

    def __init__(
        self,
        method: str = "fasterpam",
        init: str = "random",
        max_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        """Constructor for the ClusteringKMedoids class."""
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
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Run the K-Medoids clustering algorithm on a subset of samples using a distance matrix
        derived from Random Forest terminal nodes.

        The distance matrix is not precomputed in memory; it is generated on demand from
        precomputed terminal nodes, which significantly reduces memory consumption.

        :param k: Number of clusters to form.
        :type k: int
        :param distance_metric: An instance of DistanceRandomForestProximity with precomputed terminals.
        :type distance_metric: DistanceRandomForestProximity
        :param sample_indices: Indices of the samples to include in clustering.
        :type sample_indices: numpy.ndarray
        :param seed: Optional random seed for internal reproducibility (not directly used here).
        :type seed: Optional[int]

        :raises ValueError: If terminal nodes are not precomputed in the distance metric.

        :return: Cluster labels for each sample in the input.
        :rtype: numpy.ndarray
        """
        if distance_metric.terminals is None:
            raise ValueError("Terminals need to be precomputed!")
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
    """
    Implements the CLARA (Clustering Large Applications) algorithm using Random Forest proximity-based distances.
    CLARA allows K-Medoids clustering on large datasets by performing it on several subsamples of the dataset and
    selecting the best set of medoids based on total intra-cluster distances (inertia). This implementation leverages
    terminal node assignments from a trained Random Forest to compute pairwise distances as needed.

    The full distance matrix is not precomputed; instead, distances are calculated on demand for each
    subsample using precomputed terminal node indices. This approach is memory-efficient and scalable to
    large datasets.

    :param sub_sample_size: Number or proportion of samples to draw for each CLARA iteration.
    :type sub_sample_size: int
    :param sampling_iter: Number of CLARA iterations to perform. Defaults to log2(n_samples) if not specified.
    :type sampling_iter: int
    :param method: Method used in K-Medoids clustering (e.g., 'pam' or 'fasterpam').
    :type method: str
    :param init: Initialization strategy for K-Medoids (e.g., 'random' or 'build').
    :type init: str
    :param max_iter: Maximum number of iterations for the K-Medoids algorithm.
    :type max_iter: int
    :param random_state: Random seed for reproducibility.
    :type random_state: int
    """

    def __init__(
        self,
        sub_sample_size: int = None,
        sampling_iter: int = None,
        method: str = "fasterpam",
        init: str = "random",
        max_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        """Constructor for the ClusteringClara class."""
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
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Executes CLARA-based K-Medoids clustering by iteratively sampling subsets of the data,
        computing proximity-based distances on the fly using terminal node information,
        and identifying the best set of medoids based on inertia. The final clustering assigns
        labels to all samples based on the best-performing medoids.

        :param k: Number of clusters to form.
        :type k: int
        :param distance_metric: Object that contains terminal node data and computes proximity distances on the fly.
        :type distance_metric: DistanceRandomForestProximity
        :param sample_indices: Array of indices corresponding to the samples to cluster.
        :type sample_indices: numpy.ndarray
        :param seed: Optional base seed for subsampling reproducibility; overrides the default random state.
        :type seed: int or None

        :raises ValueError: If terminal nodes are not precomputed in the distance metric.

        :return: Cluster labels for each sample in the input.
        :rtype: numpy.ndarray
        """
        if distance_metric.terminals is None:
            raise ValueError("Terminals need to be precomputed!")

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

        # generate distinct seeds for each iteration
        if seed is None:
            seeds = self.random_state_.randint(0, 2**32 - 1, size=self.sampling_iter)
        else:
            rng = np.random.RandomState(seed)
            seeds = rng.randint(0, 2**32 - 1, size=self.sampling_iter)

        # iterate n times over the input dataset
        for seed in seeds:
            rng = np.random.RandomState(seed)  # individual RNG per iteration
            sub_sample_indices = rng.choice(sample_indices_mapping, size=sub_sample_size, replace=False)
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

        return cluster_labels + 1


############################################
# Numba Functions
############################################


@njit(parallel=True, fastmath=True)
def _calculate_inertia(terminals: np.ndarray, sample_idx: np.ndarray, medoids_idx: np.ndarray) -> float:
    """
    Compute the total inertia (sum of minimum distances to medoids) for a set of samples,
    where distances are derived from terminal node similarities across trees in a Random Forest.

    :param terminals: 2D array where each row corresponds to a sample and each column to the terminal node from a tree.
    :type terminals: numpy.ndarray
    :param sample_idx: Indices of samples for which inertia is calculated.
    :type sample_idx: numpy.ndarray
    :param medoids_idx: Indices of the medoid samples used to compute distances.
    :type medoids_idx: numpy.ndarray

    :return: Total inertia value for the sample set.
    :rtype: float
    """
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
def _asign_labels(terminals: np.ndarray, sample_idx: np.ndarray, medoids_idx: np.ndarray) -> np.ndarray:
    """
    Assign each sample to the cluster of the closest medoid based on Random Forest terminal node similarity.

    :param terminals: 2D array where each row corresponds to a sample and each column to the terminal node from a tree.
    :type terminals: numpy.ndarray
    :param sample_idx: Indices of the samples to assign to clusters.
    :type sample_idx: numpy.ndarray
    :param medoids_idx: Indices of the selected medoid samples used to assign cluster labels.
    :type medoids_idx: numpy.ndarray

    :return: Array of cluster labels corresponding to the closest medoid for each sample.
    :rtype: numpy.ndarray
    """
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
