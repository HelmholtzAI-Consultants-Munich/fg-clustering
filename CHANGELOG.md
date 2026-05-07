# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `DistanceRandomForestProximity.min_samples_in_node` parameter: collapses each leaf to
  the nearest ancestor whose `n_node_samples` is at least the given threshold. Reduces
  proximity-matrix sparsity for deep regression forests and produces more balanced
  clusterings. Defaults to `None` (no collapse).
- Internal helpers `_compute_parent_array`, `_build_leaf_to_ancestor_map`,
  `_validate_mutually_exclusive`, and
  `DistanceRandomForestProximity._collapse_terminals` as foundation for upcoming
  ancestor-collapse strategies.
- `DistanceRandomForestLCA`: new proximity class based on Least Common Ancestor depth
  along decision paths. Produces graded similarity that stays informative when samples
  fall into different leaves, addressing proximity sparsity for deep regression forests.
  Exposes the same duck-typed surface as `DistanceRandomForestProximity`
  (`calculate_terminals`, `calculate_distance_matrix`, `remove_distance_matrix`) plus
  new state attributes `paths` (root-to-leaf node-id paths, padded) and `path_lens`
  (per-sample per-tree effective path length).
- Internal helper `_compute_node_depths(tree)` to label every sklearn tree node with
  its depth. Reused by PR 3 (`max_depth_for_proximity`).
- Numba kernel `_calculate_lca_distances` for the LCA distance computation.
- `fgclustering` package now exports `DistanceRandomForestLCA` (added to `__all__`).
- `CHANGELOG.md` (this file).
- `DistanceRandomForestProximity.max_depth_for_proximity` parameter: collapses each
  leaf to the nearest ancestor whose tree depth is at most the given threshold,
  capping the granularity of the proximity-induced partition. `0` collapses every
  sample to the root; large values asymptote to standard terminal-node proximity.
  Defaults to `None`.
- Cross-PR infrastructure note: this parameter reuses `_compute_node_depths`
  introduced in PR 2 and `_compute_parent_array` / `_build_leaf_to_ancestor_map` /
  `DistanceRandomForestProximity._collapse_terminals` /
  `_validate_mutually_exclusive` introduced in PR 1.
- `DistanceRandomForestProximity.max_node_variance` parameter: collapses each leaf to
  the nearest ancestor whose target variance (``tree_.impurity`` under
  ``criterion="squared_error"`` or ``"friedman_mse"``) remains greater than or equal
  to the given threshold. This effectively prunes regions where the variance has
  already fallen below the threshold. For ``criterion="squared_error"``, this
  corresponds exactly to variance-based pruning; for ``criterion="friedman_mse"``,
  the behavior is an approximation, as splits are selected using Friedman's
  improvement score rather than pure variance reduction. Regression-only; requires
  a ``RandomForestRegressor`` and raises ``ValueError`` at ``calculate_terminals``
  time otherwise. Defaults to ``None``.


### Changed
- `DistanceRandomForestProximity.__init__` now accepts `min_samples_in_node` and
  validates it (must be >= 1 when not `None`).
- `DistanceRandomForestProximity.calculate_terminals` now remaps the stored terminal
  matrix to effective-ancestor ids when `min_samples_in_node` is configured. When the
  parameter is `None` (default), behavior is byte-for-byte identical to previous
  releases.
- `DistanceRandomForestLCA`: per-tree LCA similarity is now normalized by the **deeper**
  of the two leaves (``max(leaf_depth_i, leaf_depth_j)``) instead of the shallower
  (``min(...)``). Asymmetric path lengths now reduce similarity, which matches the
  intended "divergence depth relative to the longer decision path" semantic and
  prevents the prior behavior where a sample falling into a shallow leaf was treated
  as identical to any sample sharing its short prefix. The numba kernel
  ``_calculate_lca_distances`` and the ``DistanceRandomForestLCA`` class docstring
  were updated; the class API surface, stored attributes, and stored shapes are
  unchanged.
- `fgclustering/__init__.py`: expanded the `from .distance import ...` statement to
  include `DistanceRandomForestLCA`, and added it to `__all__`. No removals.
- Class and method docstrings in `fgclustering/distance.py` updated to describe the new
  effective-ancestor semantics.
- `DistanceRandomForestProximity.__init__` now accepts `max_depth_for_proximity` and
  validates it (must be a non-negative integer when not `None`). The
  `_validate_mutually_exclusive` call is extended to cover both
  `min_samples_in_node` and `max_depth_for_proximity`; setting both at once raises
  `ValueError`.
- `DistanceRandomForestProximity.calculate_terminals` dispatches to a second
  ancestor-collapse branch when `max_depth_for_proximity` is configured. Behavior
  with both new parameters set to `None` (default) is byte-for-byte identical to
  previous releases.
- `DistanceRandomForestProximity` class docstring updated to enumerate both supported
  ancestor-collapse criteria and document their mutual exclusivity.
- `DistanceRandomForestProximity.__init__` now accepts `max_node_variance` and
  validates its value range (must be >= 0 when not ``None``). The
  ``_validate_mutually_exclusive`` call is extended to cover all three
  ancestor-collapse parameters (``min_samples_in_node``,
  ``max_depth_for_proximity``, ``max_node_variance``); setting any two at once
  raises ``ValueError``.
- `DistanceRandomForestProximity.calculate_terminals` now performs an
  estimator-type guard and allows both ``criterion="squared_error"`` and
  ``criterion="friedman_mse"`` when ``max_node_variance`` is configured. A
  ``UserWarning`` is issued for ``friedman_mse`` to indicate that variance-based
  pruning is approximate in this case. Behavior with all three new parameters set
  to ``None`` (default) is byte-for-byte identical to previous releases.
- `DistanceRandomForestProximity` class docstring updated to enumerate all three
  supported ancestor-collapse criteria and document their mutual exclusivity and
  the regression-only restriction of ``max_node_variance``.

### Known limitations
- `DistanceRandomForestLCA` paired with `ClusteringClara` is not fully LCA-consistent
  end-to-end. While CLARA uses `calculate_distance_matrix` during medoid search,
  internal kernels in `fgclustering/clustering.py` still read `self.terminals`
  directly, including candidate evaluation/subsample-selection
  (`_calculate_inertia`) and final label assignment (`_asign_labels`). As a result,
  candidate medoid scoring and the final output labels are still influenced by
  terminal-node proximity rather than the LCA metric alone. `ClusteringKMedoids` is
  fully consistent with the LCA metric. Abstracting those kernels onto the distance
  class is tracked as a follow-up.
