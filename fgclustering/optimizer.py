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
from .distance import DistanceRandomForestBase
from .clustering import ClusteringKMedoids, ClusteringClara


############################################
# Optimizer Class
############################################


class Optimizer:
    """
    Optimize the number of clusters using stability and target-based quality.

    For each candidate ``k``, samples are clustered with the configured forest-derived
    distance metric and clustering strategy. The resulting clustering is evaluated by
    bootstrap-based Jaccard stability and by a task-specific quality score.

    Classification models are scored with balanced average Gini impurity. Regression models
    are scored with normalized within-cluster variation. The selected solution is the stable
    clustering with the lowest quality score.

    :param distance_metric: Forest-derived distance metric used for clustering.
    :type distance_metric: DistanceRandomForestBase
    :param clustering_strategy: Clustering strategy used to produce cluster assignments.
    :type clustering_strategy: ClusteringKMedoids | ClusteringClara
    :param random_state: Random seed used for reproducible bootstrap sampling.
    :type random_state: int | None
    """

    def __init__(
        self,
        distance_metric: DistanceRandomForestBase,
        clustering_strategy: ClusteringKMedoids | ClusteringClara,
        random_state: int | None,
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
    ) -> tuple[list[dict], int | None]:
        """
        Evaluate candidate cluster numbers and select the best stable solution.

        Each value in the inclusive ``k_range`` is clustered on the full dataset. Stability is
        estimated by repeated subsampling, reclustering each subsample, and matching bootstrap
        clusters to full-data clusters with the Jaccard index. Quality is evaluated with
        balanced average impurity for classification targets or normalized within-cluster
        variation for regression targets.

        Cluster labels are reordered by increasing mean target value before results are stored.
        The selected ``best_k`` is the stable candidate with the lowest quality score. If no
        candidate exceeds ``JI_discart_value``, ``best_k`` is ``None``.

        :param y: Target values aligned with the encoded samples.
        :type y: pd.Series
        :param k_range: Inclusive range of cluster counts to evaluate as ``(min_k, max_k)``.
        :type k_range: tuple[int, int]
        :param JI_bootstrap_iter: Number of bootstrap iterations used for stability estimation.
        :type JI_bootstrap_iter: int
        :param JI_bootstrap_sample_size: Number of samples drawn in each bootstrap iteration.
        :type JI_bootstrap_sample_size: int | float
        :param JI_discart_value: Minimum mean Jaccard index required for a clustering to be
            considered stable.
        :type JI_discart_value: float
        :param model_type: Random Forest estimator class used to select classification or
            regression scoring.
        :type model_type: type[RandomForestClassifier] | type[RandomForestRegressor]
        :param n_jobs: Number of parallel jobs used for bootstrap stability computation.
        :type n_jobs: int
        :param verbose: Verbosity level for progress bars and printed summaries.
        :type verbose: int

        :raises ValueError: If ``model_type`` is neither a Random Forest classifier nor a Random
            Forest regressor.

        :return: Tuple containing all per-``k`` result dictionaries and the selected ``best_k``.
        :rtype: tuple[list[dict], int | None]
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
            cluster_mapping = self._sort_clusters_by_target(
                y=y, cluster_labels=cluster_labels_k, model_type=model_type
            )
            reordered_cluster_labels = pd.Series(cluster_labels_k).map(cluster_mapping).to_numpy(dtype=int)
            reordered_JI_per_cluster_k = {
                cluster_mapping[cid]: JI_per_cluster_k[cid] for cid in JI_per_cluster_k.keys()
            }

            # store results
            results.append(
                {
                    "k": k_optimizer,
                    "Stable": JI_k > JI_discart_value,
                    "Mean_JI": JI_k,
                    "Score": cluster_score_k,
                    "Cluster_JI": dict(sorted(reordered_JI_per_cluster_k.items())),
                    "Cluster_labels": reordered_cluster_labels,
                }
            )

            if JI_k > JI_discart_value and cluster_score_k < best_score:
                best_k = k_optimizer
                best_score = cluster_score_k

        if verbose:
            if best_k is None:
                warnings.warn(f"No stable clusters were found for JI cutoff {JI_discart_value}!")
            if best_k is not None and best_k > 1 and not (k_range[1] - k_range[0]) == 0:
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
        Compute mean cluster-wise Jaccard stability across bootstrap iterations.

        Each bootstrap iteration reclusters a subsample and matches its clusters to the original
        full-data clustering. Jaccard scores are accumulated per original cluster and averaged
        over all bootstrap iterations.

        :param k: Number of clusters used for the original and bootstrap clusterings.
        :type k: int
        :param cluster_labels_original: Cluster labels from clustering the full dataset.
        :type cluster_labels_original: np.ndarray

        :return: Mapping from original cluster label to mean Jaccard index.
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
        Compute cluster-wise Jaccard scores for one bootstrap subsample.

        A subsample is drawn without replacement and reclustered. Bootstrap clusters are matched
        to original clusters by greedily selecting the largest remaining entries of the
        pairwise Jaccard matrix. Each original cluster receives the matched Jaccard score, or
        ``0.0`` if it is not matched.

        :param k: Number of clusters used for bootstrap clustering.
        :type k: int
        :param mapping_cluster_labels_to_samples_original: Mapping from original cluster labels
            to their sample indices.
        :type mapping_cluster_labels_to_samples_original: dict
        :param random_state_subsampling: Random seed used for this bootstrap subsample and
            clustering run.
        :type random_state_subsampling: int

        :return: Mapping from original cluster label to its bootstrap Jaccard score.
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
        Compute balanced average Gini impurity for classification clusters.

        Class counts within each cluster are weighted by inverse global class frequency before
        computing Gini impurity. This reduces dominance by frequent classes and gives rare
        classes stronger influence on the score. The final score is the mean balanced impurity
        across clusters.

        :param categorical_values: Classification target values aligned with ``cluster_labels``.
        :type categorical_values: pd.Series
        :param cluster_labels: Cluster labels for the samples.
        :type cluster_labels: np.ndarray

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
        Compute normalized within-cluster variation for regression clusters.

        Cluster-wise variances are weighted by cluster size, summed, and normalized by the total
        target variance. Lower values indicate clusters that are more homogeneous with respect
        to the continuous target.

        :param continuous_values: Regression target values aligned with ``cluster_labels``.
        :type continuous_values: pd.Series
        :param cluster_labels: Cluster labels for the samples.
        :type cluster_labels: np.ndarray

        :return: Within-cluster variation normalized by total target variation.
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
    ) -> dict[int, int]:
        """
        Rank cluster labels by increasing mean target value.

        For classification tasks, target values are converted to categorical codes before
        cluster means are computed. For regression tasks, the continuous target values are used
        directly. The returned mapping converts original cluster labels to consecutive one-based
        labels ordered by increasing mean target.

        :param y: Target values aligned with ``cluster_labels``.
        :type y: pd.Series
        :param cluster_labels: Original cluster labels.
        :type cluster_labels: np.ndarray
        :param model_type: Random Forest estimator class used to choose classification or
            regression target handling.
        :type model_type: type[RandomForestClassifier] | type[RandomForestRegressor]

        :return: Mapping from original cluster labels to reordered one-based labels.
        :rtype: dict[int, int]
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

        return cluster_mapping
