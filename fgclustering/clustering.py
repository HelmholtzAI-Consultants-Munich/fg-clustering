############################################
# Imports
############################################

import os
import kmedoids
import numpy as np

from typing import Optional, Union
from numba import njit, prange

from .distance import DistanceRandomForestProximity
from .utils import check_sub_sample_size

############################################
# Clustering Classes
############################################


class ClusteringKMedoids:
    """
    Implements the K-Medoids algorithm using Random Forest proximity-based distances.

    To ensure memory-efficiency and scalability to large datasets, a precomputed
    distance matrix is not required. Instead, the distance matrix is computed
    on the fly using Random Forest terminal nodes.

    :param method: Computation method for the K-Medoids algorithm. Default: "fasterpam".
    :type method: Optional[str]
    :param init: Initialization strategy for the K-Medoids algorithm. Default: "random".
    :type init: Optional[str]
    :param max_iter: Maximum number of iterations for the K-Medoids algorithm. Default: 100.
    :type max_iter: Optional[int]
    :param random_state: Random seed for reproducibility. Default: 42.
    :type random_state: Optional[int]
    """

    def __init__(
        self,
        method: Optional[str] = "fasterpam",
        init: Optional[str] = "random",
        max_iter: Optional[int] = 100,
        random_state: Optional[int] = 42,
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
        random_state_subsampling: Union[int, None],
        verbose: int,
    ) -> np.ndarray:
        """
        Executes K-Medoids clustering on a (sub-)set of samples by computing proximity-based
        distances on the fly using terminal node information.

        :param k: Number of clusters.
        :type k: int
        :param distance_metric: An instance of DistanceRandomForestProximity with precomputed terminals.
        :type distance_metric: DistanceRandomForestProximity
        :param sample_indices: Indices of the samples to include in clustering.
        :type sample_indices: numpy.ndarray
        :param random_state_subsampling: Random seed for for subsampling reproducibility. Not used in this implementation.
        :type random_state_subsampling: Union[int, None]
        :param verbose: Verbosity level (0 = silent, 1 = progress messages). Not used in this implementation.
        :type verbose: int

        :raises ValueError: If terminal nodes are not precomputed in the distance metric.

        :return: Cluster labels for each input sample index.
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
    Implements the CLARA (Clustering Large Applications) algorithm using Random Forest
    proximity-based distances. CLARA allows K-Medoids clustering on large datasets by
    performing it on several subsamples of the dataset and selecting the best set of
    medoids based on total intra-cluster distances (inertia).

    To optimize memory usage and enable scalability for large datasets, it is not required
    to precompute the full distance matrix. Instead, pairwise distances for subsamples are
    computed on demand from precomputed Random Forest terminal node assignments.

    :param sub_sample_size: Number or proportion of samples to draw for each CLARA iteration. If None, computes an adaptive subsample ratio based on dataset size, constrained between 10% and 80%, targeting approximately 1,000 samples. Default: None.
    :type sub_sample_size: Optional[Union[int, float]]
    :param sampling_iter: Number of CLARA iterations to perform. If None, sets the number of sampling iterations log2(sample size), with a minimum of 5 iterations to ensure sufficient sampling. Default: None.
    :type sampling_iter: Optional[int]
    :param method: Computation method for the K-Medoids algorithm. Default: "fasterpam".
    :type method: Optional[str]
    :param init: Initialization strategy for the K-Medoids algorithm. Default: "random".
    :type init: Optional[str]
    :param max_iter: Maximum number of iterations for the K-Medoids algorithm. Default: 100.
    :type max_iter: Optional[int]
    :param random_state: Random seed for reproducibility. Default: 42.
    :type random_state: Optional[int]
    """

    def __init__(
        self,
        sub_sample_size: Optional[Union[int, float]] = None,
        sampling_iter: Optional[int] = None,
        method: Optional[str] = "fasterpam",
        init: Optional[str] = "random",
        max_iter: Optional[int] = 100,
        random_state: Optional[int] = 42,
    ) -> None:
        """Constructor for the ClusteringClara class."""
        self.sub_sample_size = sub_sample_size
        self.sampling_iter = sampling_iter
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
        random_state_subsampling: Union[int, None],
        verbose: int,
    ) -> np.ndarray:
        """
        Executes CLARA-based K-Medoids clustering on a (sub-)set of samples by iteratively
        sampling subsets of the input set, computing proximity-based distances on the fly
        using terminal node information, and identifying the best set of medoids based on
        inertia. Cluster labels for all samples are assigned using the best-performing medoids.

        :param k: Number of clusters.
        :type k: int
        :param distance_metric: An instance of DistanceRandomForestProximity with precomputed terminals.
        :type distance_metric: DistanceRandomForestProximity
        :param sample_indices: Indices of the samples to include in clustering.
        :type sample_indices: numpy.ndarray
        :param random_state_subsampling: Random seed for for subsampling reproducibility. If None, use `random_state` defined in constructor.
        :type random_state_subsampling: Union[int, None]
        :param verbose: Verbosity level (0 = silent, 1 = progress messages).
        :type verbose: int

        :raises ValueError: If terminal nodes are not precomputed in the distance metric.

        :return: Cluster labels for each input sample index.
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
        sub_sample_size = check_sub_sample_size(
            sub_sample_size=self.sub_sample_size, n_samples=n_samples, verbose=verbose
        )
        if self.sampling_iter is None:
            self.sampling_iter = max(5, int(np.log2(n_samples)))

        # generate distinct seeds for each iteration
        if random_state_subsampling is None:
            rng = np.random.RandomState(self.random_state)
        else:
            rng = np.random.RandomState(random_state_subsampling)
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
                random_state=seed,
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
def _calculate_inertia(
    terminals: np.ndarray,
    sample_idx: np.ndarray,
    medoids_idx: np.ndarray,
) -> float:
    """
    Compute the total total intra-cluster distances (inertia) for a set of samples,
    where distances are defined as Random Forest proximity-based distances.

    :param terminals: 2D array of shape (n_samples, n_estimators), where each entry indicates the terminal node index for a sample in a tree.
    :type terminals: numpy.ndarray
    :param sample_idx: Indices of samples for which inertia is calculated.
    :type sample_idx: numpy.ndarray
    :param medoids_idx: Indices of medoids used to calculate inertia.
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
def _asign_labels(
    terminals: np.ndarray,
    sample_idx: np.ndarray,
    medoids_idx: np.ndarray,
) -> np.ndarray:
    """
    Assign each sample to the cluster of the closest medoid based on Random Forest proximity-based distances.

    :param terminals: 2D array of shape (n_samples, n_estimators), where each entry indicates the terminal node index for a sample in a tree.
    :type terminals: numpy.ndarray
    :param sample_idx: Indices of the samples to assign to clusters.
    :type sample_idx: numpy.ndarray
    :param medoids_idx: Indices of medoids used to assign cluster labels.
    :type medoids_idx: numpy.ndarray

    :return: Cluster labels for each input sample index.
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
