############################################
# Imports
############################################

import kmedoids
import numpy as np

from imblearn.under_sampling import RandomUnderSampler

from .distance import DistanceRandomForestBase
from .utils import check_sub_sample_size, custom_round

############################################
# Clustering Classes
############################################


class ClusteringKMedoids:
    """
    K-Medoids clustering with precomputed Random-Forest-derived distances.

    This class clusters selected samples using a distance matrix computed on demand by a
    ``DistanceRandomForestBase`` instance. The distance metric must already contain its
    forest encoding, such as terminal-node assignments for proximity distances or decision
    paths for LCA distances.

    :param method: Optimization method passed to ``kmedoids.KMedoids``.
    :type method: str
    :param init: Initialization strategy passed to ``kmedoids.KMedoids``.
    :type init: str
    :param max_iter: Maximum number of K-Medoids iterations.
    :type max_iter: int
    :param random_state: Random seed used for reproducible initialization.
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
        distance_metric: DistanceRandomForestBase,
        sample_indices: np.ndarray,
        random_state_subsampling: int | None,
        verbose: int,
    ) -> np.ndarray:
        """
        Cluster selected samples with K-Medoids.

        A pairwise distance matrix is computed for ``sample_indices`` using
        ``distance_metric`` and passed to ``kmedoids.KMedoids`` with
        ``metric="precomputed"``. After fitting, the temporary distance matrix is released.
        Returned labels use one-based indexing.

        :param k: Number of clusters.
        :type k: int
        :param distance_metric: Distance metric with a precomputed forest encoding.
        :type distance_metric: DistanceRandomForestBase
        :param sample_indices: Indices of samples to cluster.
        :type sample_indices: np.ndarray
        :param random_state_subsampling: Optional subsampling seed. Not used by this
            implementation.
        :type random_state_subsampling: int | None
        :param verbose: Verbosity level. Not used by this implementation.
        :type verbose: int
        :return: One-based cluster labels for ``sample_indices``.
        :rtype: np.ndarray
        """
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
    CLARA clustering with precomputed Random-Forest-derived distances.

    CLARA approximates K-Medoids by repeatedly clustering subsamples, evaluating each
    candidate medoid set on the full selected sample set, and retaining the medoids with the
    lowest inertia. Final labels are assigned by the nearest retained medoid.

    Distances are computed on demand by a ``DistanceRandomForestBase`` instance, which allows
    the same clustering logic to work with proximity-based and LCA-based Random Forest
    distances.

    :param sub_sample_size: Number of samples, fraction of samples, or ``None`` for an
        adaptive CLARA subsample size.
    :type sub_sample_size: int | float | None
    :param sampling_iter: Number of CLARA subsampling iterations, or ``None`` to choose a
        default based on the sample size.
    :type sampling_iter: int | None
    :param sampling_target: Optional labels used for stratified subsampling.
    :type sampling_target: list | None
    :param method: Optimization method passed to ``kmedoids.KMedoids``.
    :type method: str
    :param init: Initialization strategy passed to ``kmedoids.KMedoids``.
    :type init: str
    :param max_iter: Maximum number of K-Medoids iterations per subsample.
    :type max_iter: int
    :param random_state: Random seed used for reproducible subsampling and initialization.
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
        distance_metric: DistanceRandomForestBase,
        sample_indices: np.ndarray,
        random_state_subsampling: int | None,
        verbose: int,
    ) -> np.ndarray:
        """
        Cluster selected samples with the CLARA algorithm.

        In each CLARA iteration, a subsample of ``sample_indices`` is selected and clustered
        with K-Medoids using a precomputed distance matrix. The resulting medoids are scored
        by computing their inertia over the full selected sample set. After all iterations,
        labels are assigned to all selected samples using the best medoid set.

        If ``sampling_target`` is provided, subsamples are drawn with stratification over the
        target values corresponding to ``sample_indices``. Otherwise, samples are drawn
        uniformly without replacement.

        Returned labels use one-based indexing.

        :param k: Number of clusters.
        :type k: int
        :param distance_metric: Distance metric with a precomputed forest encoding.
        :type distance_metric: DistanceRandomForestBase
        :param sample_indices: Indices of samples to cluster.
        :type sample_indices: np.ndarray
        :param random_state_subsampling: Optional random seed for CLARA subsampling. If
            ``None``, the instance-level ``random_state`` is used.
        :type random_state_subsampling: int | None
        :param verbose: Verbosity level forwarded to subsample-size validation.
        :type verbose: int
        :return: One-based cluster labels for ``sample_indices``.
        :rtype: np.ndarray
        """
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
            sub_sample_score = distance_metric.compute_inertia(
                sample_indices, sample_indices[sub_sample_medoids_idxs]
            )

            # update if score is better
            if sub_sample_score < best_score:
                best_score = sub_sample_score
                best_medoids_idxs = sub_sample_medoids_idxs

        # assign labels to the rest of the data when best medoids are found
        cluster_labels = distance_metric.assign_labels(sample_indices, sample_indices[best_medoids_idxs])

        return cluster_labels + 1
