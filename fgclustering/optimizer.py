############################################
# Imports
############################################

import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict, Counter
from typing import Union, Tuple, Optional

from .utils import map_clusters_to_samples
from .distance import DistanceRandomForestProximity
from .clustering import ClusteringKMedoids, ClusteringClara


############################################
# Optimizer Class
############################################


class Optimizer:
    """
    This class determines the optimal number of clusters (`k`) by evaluating both the quality and stability of clustering solutions.
    This approach balances robustness and interpretability, ensuring that the chosen `k` produces reliable and meaningful subgroupings in the data.

    The optimization procedure involves the following steps:

    - Clustering: For each candidate value of `k`, clusters are generated using a Random Forest-based distance metric and a specified clustering algorithm (e.g., KMedoids or CLARA).
    - Stability Assessment: Clustering stability is estimated via bootstrapping. Repeated clustering on bootstrap samples is compared against the original clustering using the Jaccard Index.
    - Quality Evaluation: Cluster quality is measured using balanced impurity scores (for classification tasks) or intra-cluster variance (for regression tasks).
    - Selection Criterion: The optimal `k` is selected as the smallest value that yields a stable clustering with the lowest quality score.

    :param distance_metric: An instance of DistanceRandomForestProximity.
    :type distance_metric: DistanceRandomForestProximity
    :param clustering_strategy: An instance of ClusteringKMedoids or ClusteringClara.
    :type clustering_strategy: Union[ClusteringKMedoids, ClusteringClara]
    :param random_state: Random seed for reproducibility.
    :type random_state: int
    """

    def __init__(
        self,
        distance_metric: DistanceRandomForestProximity,
        clustering_strategy: Union[ClusteringKMedoids, ClusteringClara],
        random_state: int,
    ):
        """Constructor for the Optimizer class."""
        self.distance_metric = distance_metric
        self.clustering_strategy = clustering_strategy
        self.random_state = random_state

    def optimizeK(
        self,
        y: pd.Series,
        k_range: Tuple[int],
        JI_bootstrap_iter: int,
        JI_bootstrap_sample_size: Union[int, float],
        JI_discart_value: float,
        model_type: str,
        n_jobs: int,
        verbose: int,
    ) -> Tuple[int, float, dict, np.ndarray]:
        """
        Search for the optimal number of clusters within a specified range using clustering quality and stability metrics.

        This function evaluates the clustering for each candidate `k` in the given range. It selects the best k
        based on a combination of cluster stability and low impurity (classification) or low variance (regression).

        :param y: Target variable.
        :type y: Union[str, pandas.Series]
        :param k_range: Range of k values to be evaluated (min_k, max_k).
        :type k_range: Tuple[int]
        :param JI_bootstrap_iter: Number of bootstrap iterations for Jaccard Index evaluation.
        :type JI_bootstrap_iter: int
        :param JI_bootstrap_sample_size: Number of samples to draw for each JI bootstrap.
        :type JI_bootstrap_sample_size: Union[int, float]
        :param JI_discart_value: Jaccard Index threshold to discard unstable clusters.
        :type JI_discart_value: float
        :param model_type: Type of model, either "cla" for classification or "reg" for regression.
        :type model_type: str
        :param n_jobs: Number of parallel jobs.
        :type n_jobs: int
        :param verbose: Verbosity level (0 = silent, 1 = progress messages).
        :type verbose: int

        :return: Tuple containing the best k, its corresponding score, Jaccard stability per cluster, and cluster labels.
        :rtype: Tuple[int, float, dict, numpy.ndarray]
        """

        self.n_samples_original = len(y)
        self.JI_bootstrap_sample_size = JI_bootstrap_sample_size
        self.JI_bootstrap_iter = JI_bootstrap_iter
        self.n_jobs = n_jobs
        self.verbose = verbose

        k = 1
        cluster_score = np.inf
        cluster_stability = None
        cluster_labels = None
        results = []

        if self.verbose:
            print(f"Using range k = ({k_range[0]}, {k_range[1]}) to optimize k.")

        for k_optimizer in tqdm(
            range(k_range[0], k_range[1] + 1), desc="Optimizing k", disable=(self.verbose == 0)
        ):
            # compute clusters
            cluster_labels_k = self.clustering_strategy.run_clustering(
                k=k_optimizer,
                distance_metric=self.distance_metric,
                sample_indices=np.arange(self.n_samples_original),
                random_state_subsampling=None,
                verbose=self.verbose,
            )
            # compute jaccard indices
            JI_per_cluster_k = self._compute_JI(
                k=k_optimizer,
                cluster_labels_original=cluster_labels_k,
            )
            JI_k = round(np.mean([JI_per_cluster_k[cluster] for cluster in JI_per_cluster_k.keys()]), 3)

            # only continue if jaccard indices are all larger than discart_value_JI (thus all clusters are stable)
            if JI_k > JI_discart_value or (k_range[1] - k_range[0]) == 0:
                if model_type == "cla":
                    # compute balanced purities
                    cluster_score_k = self._compute_balanced_average_impurity(y, cluster_labels_k)
                elif model_type == "reg":
                    # compute the total within cluster variation
                    cluster_score_k = self._compute_total_within_cluster_variation(y, cluster_labels_k)
                if cluster_score_k < cluster_score:
                    k = k_optimizer
                    cluster_score = cluster_score_k
                    cluster_stability = JI_per_cluster_k
                    cluster_labels = cluster_labels_k
                if cluster_score_k == cluster_score:
                    JI_opt = round(
                        np.mean([cluster_stability[cluster] for cluster in cluster_stability.keys()]), 3
                    )
                    if JI_k > JI_opt:
                        k = k_optimizer
                        cluster_score = cluster_score_k
                        cluster_stability = JI_per_cluster_k
                        cluster_labels = cluster_labels_k

            else:
                cluster_score_k = None

            results.append(
                {
                    "k": k_optimizer,
                    "Stable": JI_k > JI_discart_value,
                    "Mean_JI": JI_k,
                    "Score": cluster_score_k,
                    "Cluster_JI": dict(sorted(JI_per_cluster_k.items())),
                }
            )
        if verbose:
            if k == 1:
                warnings.warn(f"No stable clusters were found for JI cutoff {JI_discart_value}!")
            if k > 1 and not (k_range[1] - k_range[0]) == 0:
                print(f"\nOptimal number of clusters k = {k}")

            results_df = pd.DataFrame(results)
            print("\nClustering Evaluation Summary:")
            print(results_df[["k", "Score", "Stable", "Mean_JI", "Cluster_JI"]].to_string(index=False))

        return k, cluster_score, cluster_stability, cluster_labels

    def _compute_JI(
        self,
        k: int,
        cluster_labels_original: np.ndarray,
    ) -> dict:
        """
        Compute the average Jaccard Index for each cluster over multiple bootstrap samples.

        :param k: Number of clusters.
        :type k: int
        :param cluster_labels_original: Cluster labels of original clustering on full dataset.
        :type cluster_labels_original: numpy.ndarray

        :return: Dictionary of average Jaccard Index per cluster.
        :rtype: dict
        """
        # generate distinct seeds for each iteration
        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 2**31 - 1, size=self.JI_bootstrap_iter)

        mapping_cluster_labels_to_samples_original = map_clusters_to_samples(cluster_labels_original)

        JI_per_cluster_bootstraps = Parallel(n_jobs=self.n_jobs)(
            delayed(self._compute_JI_single_bootstrap)(
                k=k,
                mapping_cluster_labels_to_samples_original=mapping_cluster_labels_to_samples_original,
                random_state_subsampling=seed,
            )
            for seed in seeds
        )

        JI_per_cluster_sum = defaultdict(float)
        for JI_per_cluster in JI_per_cluster_bootstraps:
            for cluster, score in JI_per_cluster.items():
                JI_per_cluster_sum[cluster] += score
        JI_per_cluster_avg = {
            int(cluster): round(float(score / self.JI_bootstrap_iter), 3)
            for cluster, score in JI_per_cluster_sum.items()
        }

        return JI_per_cluster_avg

    def _compute_JI_single_bootstrap(
        self,
        k: int,
        mapping_cluster_labels_to_samples_original: dict,
        random_state_subsampling: int,
    ) -> dict:
        """
        Compute the cluster-wise Jaccard Index for a single bootstrap sample by matching bootstraped and original clusters.

        :param k: Number of clusters.
        :type k: int
        :param mapping_cluster_labels_to_samples_original: Mapping of original cluster labels to sample indices.
        :type mapping_cluster_labels_to_samples_original: dict
        :param random_state_subsampling: Random seed for for subsampling reproducibility.
        :type random_state_subsampling: int

        :return: Dictionary of Jaccard Index per cluster.
        :rtype: dict
        """

        # individual RNG per iteration
        rng = np.random.RandomState(random_state_subsampling)
        samples = rng.choice(self.n_samples_original, size=self.JI_bootstrap_sample_size, replace=False)
        samples = np.sort(samples)

        cluster_labels_bootstrap = self.clustering_strategy.run_clustering(
            k=k,
            distance_metric=self.distance_metric,
            sample_indices=samples,
            random_state_subsampling=random_state_subsampling,
            verbose=self.verbose,
        )

        mapping_cluster_labels_to_samples_bootstrap = map_clusters_to_samples(
            cluster_labels_bootstrap, samples
        )

        clusters_original = list(mapping_cluster_labels_to_samples_original.keys())
        clusters_bootstrap = list(mapping_cluster_labels_to_samples_bootstrap.keys())

        # Determine shared sample set (only those present in bootstrap)
        samples_bootstrap_all = set().union(*mapping_cluster_labels_to_samples_bootstrap.values())

        # Filter original samples to those present in bootstrap
        mapping_cluster_labels_to_samples_original_filtered = {
            label: samples.intersection(samples_bootstrap_all)
            for label, samples in mapping_cluster_labels_to_samples_original.items()
        }

        # Initialize Jaccard matrix
        jaccard_matrix = np.zeros((len(clusters_original), len(clusters_bootstrap)))

        for i, label_original in enumerate(clusters_original):
            samples_original = np.array(
                list(mapping_cluster_labels_to_samples_original_filtered[label_original])
            )
            for j, label_bootstrap in enumerate(clusters_bootstrap):
                indices_bootstrap = np.array(
                    list(mapping_cluster_labels_to_samples_bootstrap[label_bootstrap])
                )
                intersection = np.intersect1d(samples_original, indices_bootstrap, assume_unique=True)
                union = np.union1d(samples_original, indices_bootstrap)
                jaccard_matrix[i, j] = len(intersection) / len(union)

        # Map original cluster to best Jaccard index using greedy assignment
        JI_per_cluster = {label: 0.0 for label in clusters_original}

        for _ in range(min(len(clusters_original), len(clusters_bootstrap))):
            max_idx = np.argmax(jaccard_matrix)
            i, j = divmod(max_idx, jaccard_matrix.shape[1])
            JI_per_cluster[clusters_original[i]] = jaccard_matrix[i, j]
            jaccard_matrix[i, :] = -np.inf
            jaccard_matrix[:, j] = -np.inf

        return JI_per_cluster

    def _compute_balanced_average_impurity(
        self,
        categorical_values: pd.Series,
        cluster_labels: np.ndarray,
    ) -> float:
        """
        Compute the balanced Gini impurity across clusters for classification tasks.

        :param categorical_values: Target labels for each sample.
        :type categorical_values: pandas.Series
        :param cluster_labels: Cluster assignments for each sample.
        :type cluster_labels: numpy.ndarray

        :return: Mean balanced Gini impurity across clusters.
        :rtype: float
        """

        unique_classes = np.unique(categorical_values)
        unique_clusters = np.unique(cluster_labels)

        # compute the number of datapoints for each class to use it then for rescaling of the
        # class sizes within each cluster --> rescaling with inverse class size
        class_counts = Counter(categorical_values)
        rescaling_factor = {cls: 1 / class_counts[cls] for cls in unique_classes}

        score_sum = 0.0

        for cluster in unique_clusters:
            categorical_values_cluster = categorical_values[cluster_labels == cluster]

            # Count class occurrences in the cluster
            cluster_class_counts = Counter(categorical_values_cluster)

            # Rescaled class probabilities
            class_probabilities_unnormalized = np.array(
                [cluster_class_counts.get(cls, 0) * rescaling_factor[cls] for cls in unique_classes]
            )

            class_probabilities = class_probabilities_unnormalized / class_probabilities_unnormalized.sum()

            # compute (balanced) gini impurity
            gini_impurity = 1 - np.sum(class_probabilities**2)
            score_sum += gini_impurity

        return round(float(score_sum / len(unique_clusters)), 6)

    def _compute_total_within_cluster_variation(
        self,
        continuous_values: pd.Series,
        cluster_labels: np.ndarray,
    ) -> float:
        """
        Compute total within-cluster variation relative to the overall variance, used for regression evaluation.

        :param continuous_values: Target values for each sample.
        :type continuous_values: pandas.Series
        :param cluster_labels: Cluster assignments for each sample.
        :type cluster_labels: numpy.ndarray

        :return: Normalized within-cluster variation score.
        :rtype: float
        """

        total_variance = np.var(continuous_values) * len(continuous_values)
        if total_variance == 0:
            return 0.0

        within_variance = 0.0
        for cluster in np.unique(cluster_labels):
            continuous_values_cluster = continuous_values[cluster_labels == cluster]
            within_variance += np.var(continuous_values_cluster) * len(continuous_values_cluster)

        return round(float(within_variance / total_variance), 6)
