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
