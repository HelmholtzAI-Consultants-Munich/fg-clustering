############################################
# Imports
############################################

import os
import gc
import kmedoids
import numpy as np

from numba import njit, prange

from imblearn.under_sampling import RandomUnderSampler

from .distance import DistanceRandomForestProximity
from .utils import check_sub_sample_size, custom_round

############################################
# Clustering Classes
############################################


class ClusteringKMedoids:
    """
    Cluster samples with the K-Medoids algorithm using Random Forest proximity-based distances.

    Pairwise distances are derived from precomputed Random Forest terminal node
    assignments. The distance matrix for the requested samples is computed on demand,
    which avoids requiring a permanently stored full distance matrix and supports
    memory-efficient workflows.

    :param method: Optimization method used by the K-Medoids implementation.
    :type method: str
    :param init: Initialization strategy used for selecting starting medoids.
    :type init: str
    :param max_iter: Maximum number of K-Medoids update iterations.
    :type max_iter: int
    :param random_state: Random seed used for reproducibility.
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
        random_state_subsampling: int | None,
        verbose: int,
    ) -> np.ndarray:
        """
        Run K-Medoids clustering on the selected samples using Random Forest proximity distances.

        The required distance matrix is computed from the precomputed terminal node
        assignments stored in ``distance_metric``. The resulting labels are shifted to
        one-based indexing before being returned.

        :param k: Number of clusters.
        :type k: int
        :param distance_metric: Distance metric object with precomputed Random Forest terminal nodes.
        :type distance_metric: DistanceRandomForestProximity
        :param sample_indices: Indices of the samples to cluster.
        :type sample_indices: np.ndarray
        :param random_state_subsampling: Optional subsampling seed. Not used in this implementation.
        :type random_state_subsampling: int | None
        :param verbose: Verbosity level. Not used in this implementation.
        :type verbose: int

        :raises ValueError: If terminal nodes have not been precomputed in ``distance_metric``.

        :return: One-based cluster labels for the selected samples.
        :rtype: np.ndarray
        """
        if distance_metric.terminals is None:
            raise ValueError("Terminals need to be precomputed!")
        distance_matrix, file = distance_metric.calculate_distance_matrix(sample_indices=sample_indices)

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

        distance_metric.remove_distance_matrix(distance_matrix, file)

        return cluster_labels + 1


class ClusteringClara:
    """
    Cluster samples with the CLARA algorithm using Random Forest proximity-based distances.

    CLARA repeatedly draws subsamples, runs K-Medoids on each subsample, and selects the
    medoid set with the best total inertia on the full input sample set. Distances are
    computed on demand from precomputed Random Forest terminal node assignments, which
    enables scalable clustering without requiring a full precomputed distance matrix in
    memory.

    :param sub_sample_size: Number or proportion of samples drawn in each CLARA iteration, or ``None`` to choose an adaptive size.
    :type sub_sample_size: int | float | None
    :param sampling_iter: Number of CLARA subsampling iterations, or ``None`` to choose it automatically.
    :type sampling_iter: int | None
    :param sampling_target: Optional target values used for stratified subsampling.
    :type sampling_target: list | None
    :param method: Optimization method used by the K-Medoids implementation.
    :type method: str
    :param init: Initialization strategy used for selecting starting medoids.
    :type init: str
    :param max_iter: Maximum number of K-Medoids update iterations.
    :type max_iter: int
    :param random_state: Random seed used for reproducibility.
    :type random_state: int
    """

    def __init__(
        self,
        sub_sample_size: int | float | None = None,
        sampling_iter: int | None = None,
        sampling_target: list | None = None,
        method: str = "fasterpam",
        init: str = "random",
        max_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        """Constructor for the ClusteringClara class."""
        self.sub_sample_size = sub_sample_size
        self.sampling_iter = sampling_iter
        self.sampling_target = sampling_target
        if self.sampling_target is not None:
            self.sampling_target = np.array(self.sampling_target)
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
        random_state_subsampling: int | None,
        verbose: int,
    ) -> np.ndarray:
        """
        Run CLARA clustering on the selected samples using Random Forest proximity distances.

        Repeated subsamples of the input sample set are drawn, K-Medoids is fit on each
        subsample, and the medoid set with the lowest full-sample inertia is retained. Final
        labels for all selected samples are then assigned according to the best medoids. The
        returned labels use one-based indexing.

        :param k: Number of clusters.
        :type k: int
        :param distance_metric: Distance metric object with precomputed Random Forest terminal nodes.
        :type distance_metric: DistanceRandomForestProximity
        :param sample_indices: Indices of the samples to cluster.
        :type sample_indices: np.ndarray
        :param random_state_subsampling: Optional random seed controlling CLARA subsampling; if ``None``, the instance-level ``random_state`` is used.
        :type random_state_subsampling: int | None
        :param verbose: Verbosity level controlling progress-related output from helper routines.
        :type verbose: int

        :raises ValueError: If terminal nodes have not been precomputed in ``distance_metric``.

        :return: One-based cluster labels for the selected samples.
        :rtype: np.ndarray
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
            sub_sample_size=self.sub_sample_size,
            n_samples=n_samples,
            application="CLARA algorithm",
            verbose=verbose,
        )
        if self.sampling_iter is None:
            self.sampling_iter = max(5, int(np.log2(n_samples)))

        # generate distinct seeds for each iteration
        if random_state_subsampling is None:
            rng = np.random.RandomState(self.random_state)
        else:
            rng = np.random.RandomState(random_state_subsampling)
        seeds = rng.randint(0, 2**31 - 1, size=self.sampling_iter)

        # iterate n times over the input dataset
        for i, seed in enumerate(seeds):
            if self.sampling_target is not None:
                sampling_target = self.sampling_target[sample_indices]
                sub_sample_fraction = sub_sample_size / n_samples

                unique_values, unique_values_counts = np.unique(sampling_target, return_counts=True)
                sampling_strategy = {
                    int(value): custom_round(value_count * sub_sample_fraction)
                    for value, value_count in zip(unique_values, unique_values_counts)
                }
                rus = RandomUnderSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=seed,
                    replacement=False,
                )
                sub_sample_indices, _ = rus.fit_resample(
                    sample_indices_mapping.reshape(-1, 1), sampling_target
                )
                sub_sample_indices = sub_sample_indices.reshape(-1)
            else:
                rng = np.random.RandomState(seed)  # individual RNG per iteration
                sub_sample_indices = rng.choice(sample_indices_mapping, size=sub_sample_size, replace=False)
                sub_sample_indices = np.sort(sub_sample_indices)

            # distance matrix for subsample but input original indices
            sub_sample_distance_matrix, file = distance_metric.calculate_distance_matrix(
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
            distance_metric.remove_distance_matrix(sub_sample_distance_matrix, file)

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


@njit(parallel=True)
def _calculate_inertia(
    terminals: np.ndarray,
    sample_idx: np.ndarray,
    medoids_idx: np.ndarray,
) -> float:
    """
    Compute the total inertia of a sample set with respect to a given set of medoids.

    For each sample, the distance to the closest medoid is computed from Random Forest
    terminal node proximity, and these minimum distances are summed across all samples.

    :param terminals: Array of terminal node assignments with shape ``(n_samples, n_estimators)``.
    :type terminals: np.ndarray
    :param sample_idx: Indices of the samples whose inertia is evaluated.
    :type sample_idx: np.ndarray
    :param medoids_idx: Indices of the medoid samples.
    :type medoids_idx: np.ndarray

    :return: Total inertia of the sample set.
    :rtype: float
    """
    n_estimators = terminals.shape[1]
    n_samples = len(sample_idx)
    n_medoids = len(medoids_idx)
    inertia = np.zeros(n_samples)

    for i in prange(n_samples):
        sample = sample_idx[i]
        distances = np.empty(n_medoids, dtype=np.float32)

        for j in range(n_medoids):
            medoid = medoids_idx[j]
            # use explicit loop for proximity to avoid temporary array allocation and minimize memory traffic
            proximity = 0
            for t in range(n_estimators):
                if terminals[sample, t] == terminals[medoid, t]:
                    proximity += 1
            distances[j] = 1.0 - (proximity / n_estimators)

        inertia[i] = np.min(distances)

    return np.sum(inertia)


@njit(parallel=True)
def _asign_labels(
    terminals: np.ndarray,
    sample_idx: np.ndarray,
    medoids_idx: np.ndarray,
) -> np.ndarray:
    """
    Assign each sample to the nearest medoid using Random Forest proximity-based distances.

    For every sample in ``sample_idx``, the distance to each medoid is computed from the
    terminal node assignments, and the label of the closest medoid is returned using
    zero-based indexing.

    :param terminals: Array of terminal node assignments with shape ``(n_samples, n_estimators)``.
    :type terminals: np.ndarray
    :param sample_idx: Indices of the samples to assign.
    :type sample_idx: np.ndarray
    :param medoids_idx: Indices of the medoid samples.
    :type medoids_idx: np.ndarray

    :return: Zero-based cluster labels for the input samples.
    :rtype: np.ndarray
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
            # use explicit loop for proximity to avoid temporary array allocation and minimize memory traffic
            proximity = 0
            for t in range(n_estimators):
                if terminals[sample, t] == terminals[medoid, t]:
                    proximity += 1
            cluster_label_sample[j] = 1.0 - (proximity / n_estimators)

        cluster_labels[i] = np.argmin(cluster_label_sample)

    return cluster_labels
