############################################
# Imports
############################################

import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict, Counter

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .utils import map_clusters_to_samples
from .distance import DistanceRandomForestProximity
from .clustering import ClusteringKMedoids, ClusteringClara


############################################
# Optimizer Class
############################################


class Optimizer:
    """
    Determine the optimal number of clusters by jointly evaluating clustering quality and stability.

    For each candidate value of ``k``, clusters are generated using a Random Forest-based
    distance metric and the chosen clustering strategy. Stability is estimated through
    repeated subsampling and comparison to the original clustering using the Jaccard
    Index. Cluster quality is evaluated with balanced impurity for classification tasks
    or within-cluster variation for regression tasks. The selected solution is the
    stable clustering with the best quality score.

    :param distance_metric: Distance metric based on Random Forest proximity.
    :type distance_metric: DistanceRandomForestProximity
    :param clustering_strategy: Clustering strategy used to generate cluster assignments.
    :type clustering_strategy: ClusteringKMedoids | ClusteringClara
    :param random_state: Random seed used for reproducibility.
    :type random_state: int
    """

    def __init__(
        self,
        distance_metric: DistanceRandomForestProximity,
        clustering_strategy: ClusteringKMedoids | ClusteringClara,
        random_state: int,
    ):
        """Constructor for the Optimizer class."""
        self.distance_metric = distance_metric
        self.clustering_strategy = clustering_strategy
        self.random_state = random_state

    def optimizeK(
        self,
        y: pd.Series,
        k_range: tuple[int, int],
        JI_bootstrap_iter: int,
        JI_bootstrap_sample_size: int | float,
        JI_discart_value: float,
        model_type: type[RandomForestClassifier] | type[RandomForestRegressor],
        n_jobs: int,
        verbose: int,
    ) -> tuple[list[dict], int]:
        """
        Search for the optimal number of clusters within a given range using quality and stability criteria.

        For each value of ``k`` in ``k_range``, clustering is performed on the full dataset,
        cluster stability is estimated using repeated subsampling and Jaccard Index matching,
        and a task-specific cluster quality score is computed. Classification tasks use
        balanced average impurity, whereas regression tasks use normalized within-cluster
        variation. The best solution is the stable clustering with the lowest score.

        :param y: Target values aligned with the full dataset.
        :type y: pd.Series
        :param k_range: Inclusive range of cluster counts to evaluate, given as ``(min_k, max_k)``.
        :type k_range: tuple[int, int]
        :param JI_bootstrap_iter: Number of subsampling iterations used to estimate Jaccard stability.
        :type JI_bootstrap_iter: int
        :param JI_bootstrap_sample_size: Number or fraction of samples drawn in each stability iteration.
        :type JI_bootstrap_sample_size: int | float
        :param JI_discart_value: Minimum mean Jaccard Index required for a clustering to be considered stable.
        :type JI_discart_value: float
        :param model_type: Estimator class used to determine whether classification or regression scoring is applied.
        :type model_type: type[RandomForestClassifier] | type[RandomForestRegressor]
        :param n_jobs: Number of parallel jobs used for the stability computation.
        :type n_jobs: int
        :param verbose: Verbosity level controlling progress output and printed summaries.
        :type verbose: int

        :return: List of result dictionaries for all evaluated ``k`` values and the selected best ``k``.
        :rtype: tuple[list[dict], int]
        """

        self.n_samples_original = len(y)
        self.JI_bootstrap_sample_size = JI_bootstrap_sample_size
        self.JI_bootstrap_iter = JI_bootstrap_iter
        self.n_jobs = n_jobs
        self.verbose = verbose

        best_k = None
        best_score = np.inf
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

            # compute cluster score
            if issubclass(model_type, RandomForestClassifier):
                # compute balanced purities
                cluster_score_k = self._compute_balanced_average_impurity(y, cluster_labels_k)
            elif issubclass(model_type, RandomForestRegressor):
                # compute the total within cluster variation
                cluster_score_k = self._compute_total_within_cluster_variation(y, cluster_labels_k)
            else:
                raise ValueError(
                    "model_type must be RandomForestClassifier or RandomForestRegressor (or a subclass thereof)."
                )

            # reorder cluster labels by target mean
            reordered_cluster_labels = self._sort_clusters_by_target(
                y=y, cluster_labels=cluster_labels_k, model_type=model_type
            )

            # store results
            results.append(
                {
                    "k": k_optimizer,
                    "Stable": JI_k > JI_discart_value,
                    "Mean_JI": JI_k,
                    "Score": cluster_score_k,
                    "Cluster_JI": dict(sorted(JI_per_cluster_k.items())),
                    "Cluster_labels": reordered_cluster_labels,
                }
            )

            if JI_k > JI_discart_value and cluster_score_k < best_score:
                best_k = k_optimizer
                best_score = cluster_score_k

        if verbose:
            if best_k == 1:
                warnings.warn(f"No stable clusters were found for JI cutoff {JI_discart_value}!")
            if best_k > 1 and not (k_range[1] - k_range[0]) == 0:
                print(f"\nOptimal number of clusters k = {best_k}")

            print("\nClustering Evaluation Summary:")
            print(
                pd.DataFrame(results)[["k", "Score", "Stable", "Mean_JI", "Cluster_JI"]].to_string(
                    index=False
                )
            )

        return results, best_k

    def _compute_JI(
        self,
        k: int,
        cluster_labels_original: np.ndarray,
    ) -> dict:
        """
        Compute the average cluster-wise Jaccard Index over repeated subsampling runs.

        For each subsampling iteration, a clustering is computed on the sampled observations
        and matched to the original clustering. Jaccard scores are averaged across all
        iterations for each original cluster.

        :param k: Number of clusters used in the clustering solution.
        :type k: int
        :param cluster_labels_original: Cluster labels obtained from clustering the full dataset.
        :type cluster_labels_original: np.ndarray

        :return: Dictionary mapping each original cluster label to its average Jaccard Index.
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
        Compute cluster-wise Jaccard Index values for a single subsampling iteration.

        A subset of samples is drawn without replacement, clustered independently, and then
        matched to the original clustering using greedy assignment on the pairwise Jaccard
        overlap matrix. Each original cluster is assigned the best available matching score.

        :param k: Number of clusters used in the clustering solution.
        :type k: int
        :param mapping_cluster_labels_to_samples_original: Mapping from original cluster labels to sample indices.
        :type mapping_cluster_labels_to_samples_original: dict
        :param random_state_subsampling: Random seed controlling the sampled observations and clustering reproducibility.
        :type random_state_subsampling: int

        :return: Dictionary mapping original cluster labels to Jaccard Index values for this iteration.
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
                intersection_size = np.intersect1d(
                    samples_original, indices_bootstrap, assume_unique=True
                ).size
                union_size = samples_original.size + indices_bootstrap.size - intersection_size
                jaccard_matrix[i, j] = intersection_size / union_size

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
        Compute the balanced average Gini impurity across clusters for classification tasks.

        Class frequencies are reweighted by the inverse global class frequency so that rare
        classes contribute proportionally more. For each cluster, a balanced class
        distribution is computed and converted to Gini impurity. The final score is the mean
        impurity across all clusters.

        :param categorical_values: Categorical target values for each sample.
        :type categorical_values: pd.Series
        :param cluster_labels: Cluster assignment for each sample.
        :type cluster_labels: np.ndarray

        :return: Mean balanced Gini impurity across all clusters.
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
        Compute normalized within-cluster variation for regression tasks.

        The total within-cluster variance is computed as the sum of cluster-wise variances
        weighted by cluster size and then normalized by the total variance of the full target
        vector. Lower values indicate more homogeneous clusters with respect to the target.

        :param continuous_values: Continuous target values for each sample.
        :type continuous_values: pd.Series
        :param cluster_labels: Cluster assignment for each sample.
        :type cluster_labels: np.ndarray

        :return: Within-cluster variation normalized by the total variance.
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

    def _sort_clusters_by_target(
        self,
        y: pd.Series,
        cluster_labels: np.ndarray,
        model_type: type[RandomForestClassifier] | type[RandomForestRegressor],
    ) -> np.ndarray:
        """
        Reorder cluster labels according to the mean target value within each cluster.

        For classification tasks, target values are first converted to category codes. Cluster
        means are then computed and used to rank clusters in ascending order. Original
        cluster labels are remapped to consecutive labels starting at 1.

        :param y: Target values aligned with the cluster labels.
        :type y: pd.Series
        :param cluster_labels: Cluster labels produced by forest-guided clustering.
        :type cluster_labels: np.ndarray
        :param model_type: Estimator class used to determine whether classification or regression handling is applied.
        :type model_type: type[RandomForestClassifier] | type[RandomForestRegressor]

        :return: Cluster labels remapped according to ascending mean target value.
        :rtype: np.ndarray
        """

        # ensure y is a series
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # use category codes for classification
        target = y.astype("category").cat.codes if issubclass(model_type, RandomForestClassifier) else y

        df = pd.DataFrame({"cluster": cluster_labels, "target": target})

        # get the mean target for each cluster
        mean_per_cluster = df.groupby("cluster")["target"].mean()

        # get the original cluster labels, sorted by their mean target
        sorted_clusters = mean_per_cluster.sort_values().index

        # create mapping from the old cluster labels to the new, ranked labels
        cluster_mapping = {
            old_label: new_label for new_label, old_label in enumerate(sorted_clusters, start=1)
        }

        # apply the re-labeling
        new_labels = pd.Series(cluster_labels).map(cluster_mapping)

        return new_labels.to_numpy()
