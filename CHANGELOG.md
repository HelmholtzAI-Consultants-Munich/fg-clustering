# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `DistanceRandomForestProximity.min_samples_in_node` parameter: collapses each leaf to
  the nearest ancestor whose ``n_node_samples`` is at least the given threshold. Reduces
  proximity-matrix sparsity for deep regression forests and produces more balanced
  clusterings. Defaults to `None` (no collapse).
- Internal helpers `_compute_parent_array`, `_build_leaf_to_ancestor_map`,
  `_validate_mutually_exclusive`, and
  `DistanceRandomForestProximity._collapse_terminals` as foundation for upcoming
  ancestor-collapse strategies.
- `CHANGELOG.md` (this file).

### Changed
- `DistanceRandomForestProximity.__init__` now accepts `min_samples_in_node` and
  validates it (must be >= 1 when not `None`).
- `DistanceRandomForestProximity.calculate_terminals` now remaps the stored terminal
  matrix to effective-ancestor ids when `min_samples_in_node` is configured. When the
  parameter is `None` (default), behavior is byte-for-byte identical to previous
  releases.
- Class and method docstrings in `fgclustering/distance.py` updated to describe the new
  effective-ancestor semantics.
