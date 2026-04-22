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
from typing import Callable

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
    Compute a proximity-based distance matrix from the terminal nodes of a trained Random Forest model,
    or from a coarser inner-node projection when an ancestor-collapse criterion is provided.

    Sample similarity is derived from how often two samples fall into the same terminal node
    across trees. Distances are defined as one minus this proximity. Optionally, leaves can be
    collapsed to a coarser ancestor according to a structural criterion (``min_samples_in_node``),
    which reduces proximity sparsity for deep regression forests. The distance matrix can be
    computed fully in memory or stored in a disk-backed memmap array for memory-efficient operation.

    :param memory_efficient: Whether to store the distance matrix in a disk-backed memmap array.
    :type memory_efficient: bool
    :param dir_distance_matrix: Directory used to store the memmap distance matrix when ``memory_efficient=True``.
    :type dir_distance_matrix: str | None
    :param min_samples_in_node: Minimum ``n_node_samples`` required for a node to be used as
        an effective leaf. Each leaf is replaced by the nearest ancestor that has at least this
        many training samples (falling back to the root if no ancestor satisfies the criterion).
        Defaults to ``None``.
    :type min_samples_in_node: int | None
    """

    def __init__(
        self,
        memory_efficient: bool = False,
        dir_distance_matrix: str | None = None,
        min_samples_in_node: int | None = None,
    ) -> None:
        """Constructor for the DistanceRandomForestProximity class."""
        if memory_efficient:
            if dir_distance_matrix is None:
                raise ValueError("You must specify `dir_distance_matrix` when `memory_efficient=True`.")

        if min_samples_in_node is not None and (
            isinstance(min_samples_in_node, bool)
            or not isinstance(min_samples_in_node, int)
            or min_samples_in_node < 1
        ):
            raise ValueError("`min_samples_in_node` must be a positive integer.")

        _validate_mutually_exclusive(min_samples_in_node=min_samples_in_node)

        self.terminals: np.ndarray | None = None
        self.memory_efficient = memory_efficient
        self.dir_distance_matrix = dir_distance_matrix
        self.min_samples_in_node = min_samples_in_node
        self.precomputed_distance_matrix = None

    def calculate_terminals(
        self,
        estimator: RandomForestClassifier | RandomForestRegressor,
        X: pd.DataFrame,
    ) -> None:
        """
        Compute and store the terminal-node (or effective-ancestor) assignments of all samples across all trees.

        Terminal node IDs are obtained by applying the trained Random Forest to ``X``. When an
        ancestor-collapse criterion is configured on the class (e.g. ``min_samples_in_node``), each
        terminal id is replaced by the id of the nearest ancestor that satisfies the criterion.
        Each row of the stored matrix corresponds to a sample and each column to a tree.

        :param estimator: Trained Random Forest estimator.
        :type estimator: RandomForestClassifier | RandomForestRegressor
        :param X: Input feature matrix.
        :type X: pd.DataFrame

        :return: ``None``
        :rtype: None
        """
        self.terminals = estimator.apply(X).astype(np.int32)

        if self.min_samples_in_node is not None:
            min_samples = self.min_samples_in_node
            self.terminals = self._collapse_terminals(
                estimator=estimator,
                predicate_factory=lambda tree: (lambda node: tree.n_node_samples[node] >= min_samples),
            )

    def _collapse_terminals(
        self,
        estimator: RandomForestClassifier | RandomForestRegressor,
        predicate_factory: Callable,
    ) -> np.ndarray:
        """
        Collapse terminal-node assignments to coarser ancestors using a per-tree predicate.

        For each tree in the forest, a leaf-to-ancestor map is built with
        :func:`_build_leaf_to_ancestor_map` using the predicate returned by ``predicate_factory``.
        The stored terminal matrix is then remapped column by column. Requires
        :meth:`calculate_terminals` to have been run first.

        :param estimator: Trained Random Forest estimator whose ``estimators_`` are traversed.
        :type estimator: RandomForestClassifier | RandomForestRegressor
        :param predicate_factory: Callable ``tree -> (node_id -> bool)``.
        :type predicate_factory: Callable

        :return: Remapped terminal matrix of shape ``(n_samples, n_estimators)`` as ``int32``.
        :rtype: np.ndarray
        """
        if self.terminals is None:
            raise ValueError(
                "No precomputed terminals available to collapse! Run `calculate_terminals()` first."
            )

        terminals = self.terminals
        collapsed = np.empty_like(terminals)
        for t, dt in enumerate(estimator.estimators_):
            tree = dt.tree_
            parent = _compute_parent_array(tree)
            predicate = predicate_factory(tree)
            leaf_map = _build_leaf_to_ancestor_map(tree, parent, predicate)
            collapsed[:, t] = leaf_map[terminals[:, t]]
        return collapsed

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
            distance_matrix: np.ndarray | np.memmap

            if self.memory_efficient:
                if self.dir_distance_matrix is None:
                    raise ValueError("You must specify `dir_distance_matrix` when `memory_efficient=True`.")
                buffer_factor = 1.2  # 20% safety buffer
                required_bytes = int(n * n * 4 * buffer_factor)  # float32 = 4 bytes
                if not check_disk_space(self.dir_distance_matrix, required_bytes):
                    raise MemoryError(
                        f"Not enough free space to allocate a {required_bytes / 1e9:.2f} GB memmap distance matrix (with 20% buffer)."
                    )
                file_distance_matrix = os.path.join(
                    self.dir_distance_matrix,
                    f"distance_matrix_{uuid.uuid4().hex[:8]}.dat",
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
        if isinstance(distance_matrix, np.memmap):
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
# Tree helpers
############################################


def _compute_parent_array(tree) -> np.ndarray:
    """
    Compute the parent-node index for each node in a sklearn decision tree.

    The root node has parent ``-1``. Entries for unreachable nodes (if any) remain
    ``-1``. The array is built with a single pass over the ``children_left`` and
    ``children_right`` arrays of the tree.

    :param tree: Underlying sklearn ``Tree`` object (``estimator.estimators_[t].tree_``).
    :type tree: sklearn.tree._tree.Tree

    :return: Array of shape ``(n_nodes,)`` with the parent index of each node; root is ``-1``.
    :rtype: np.ndarray
    """
    n_nodes = tree.node_count
    parent = np.full(n_nodes, -1, dtype=np.int32)
    children_left = tree.children_left
    children_right = tree.children_right
    for node in range(n_nodes):
        left = children_left[node]
        right = children_right[node]
        if left != -1:
            parent[left] = node
        if right != -1:
            parent[right] = node
    return parent


def _build_leaf_to_ancestor_map(tree, parent: np.ndarray, predicate) -> np.ndarray:
    """
    Build a node-id -> effective-ancestor-id map by walking each leaf up to the
    nearest ancestor that satisfies ``predicate``.

    For every leaf node in the tree, walk up the parent chain until ``predicate(node)``
    returns ``True`` or the root is reached. The answer is memoized on intermediate
    nodes during the walk, giving amortized linear cost over all leaves. The returned
    array is indexed by original node id; only entries corresponding to actual leaves
    are consumed downstream.

    :param tree: Underlying sklearn ``Tree`` object.
    :type tree: sklearn.tree._tree.Tree
    :param parent: Parent-node array produced by :func:`_compute_parent_array`.
    :type parent: np.ndarray
    :param predicate: Callable ``node_id -> bool`` evaluated on internal and leaf nodes.
    :type predicate: Callable[[int], bool]

    :return: Array of shape ``(n_nodes,)`` mapping each leaf to its effective ancestor id.
    :rtype: np.ndarray
    """
    n_nodes = tree.node_count
    resolved = np.full(n_nodes, -1, dtype=np.int32)
    children_left = tree.children_left

    for node in range(n_nodes):
        if children_left[node] != -1:
            continue

        path = []
        cur = node
        while resolved[cur] == -1 and not predicate(cur):
            path.append(cur)
            nxt = parent[cur]
            if nxt == -1:
                break
            cur = nxt

        ancestor = resolved[cur] if resolved[cur] != -1 else cur
        for path_node in path:
            resolved[path_node] = ancestor
        resolved[cur] = ancestor

    return resolved


def _validate_mutually_exclusive(**named_params) -> None:
    """
    Raise ``ValueError`` if more than one of the passed parameters is not ``None``.

    :param named_params: Keyword arguments to check for mutual exclusivity.

    :raises ValueError: If two or more arguments are not ``None``.

    :return: ``None``
    :rtype: None
    """
    set_params = [name for name, value in named_params.items() if value is not None]
    if len(set_params) > 1:
        raise ValueError(f"Parameters {set_params} are mutually exclusive; only one may be set.")


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
