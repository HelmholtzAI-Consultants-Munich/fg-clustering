# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `DistanceRandomForestBase.compute_inertia(sample_idx, medoids_idx)` and
  `DistanceRandomForestBase.assign_labels(sample_idx, medoids_idx)`:
  distance-class methods used by `ClusteringClara` for medoid-search inertia
  and final label assignment. `DistanceRandomForestLCA` overrides both with
  LCA-aware numba kernels.
- New numba kernels `_calculate_inertia_lca` and `_assign_labels_lca` (LCA-aware
  versions of the existing terminal-node-equality kernels).
- Documentation coverage in README, Sphinx (`basic_usage.rst` via the README include),
  and tutorial notebook `special_case_tree_pruning_with_FGC` for all three
  `DistanceRandomForestProximity` collapse strategies and
  `DistanceRandomForestLCA`.
- Extended `test/integration_tests.ipynb` with examples for inner-node proximity
  collapse modes and LCA distance.
- Regression tests ``test_clara_with_lca_distance_uses_lca_inertia`` and
  ``test_compute_inertia_consistency_with_distance_matrix`` in
  ``test/test_clustering.py``.
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
  (`calculate_forest_encoding`, `calculate_distance_matrix`, `remove_distance_matrix`) plus
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
- `DistanceRandomForestProximity.min_variance_in_node` parameter: collapses each leaf to
  the nearest ancestor whose target variance (``tree_.impurity`` under
  ``criterion="squared_error"`` or ``"friedman_mse"``) remains greater than or equal
  to the given threshold. This effectively prunes regions where the variance has
  already fallen below the threshold. For ``criterion="squared_error"``, this
  corresponds exactly to variance-based pruning; for ``criterion="friedman_mse"``,
  the behavior is an approximation, as splits are selected using Friedman's
  improvement score rather than pure variance reduction. Regression-only; requires
  a ``RandomForestRegressor`` and raises ``ValueError`` at
  ``calculate_forest_encoding`` time otherwise. Defaults to ``None``.


### Changed
- **Breaking:** ``calculate_terminals`` renamed to ``calculate_forest_encoding`` on
  ``DistanceRandomForestBase``, ``DistanceRandomForestProximity``, and
  ``DistanceRandomForestLCA``. Tests, profiling scripts, tutorials, and
  ``forest_guided_clustering()`` call sites updated accordingly.
- README installation and usage sections streamlined.
- Class and method docstrings expanded across ``fgclustering/distance.py``,
  ``fgclustering/clustering.py``, ``fgclustering/forest_guided_clustering.py``,
  ``fgclustering/optimizer.py``, ``fgclustering/statistics.py``, and
  ``fgclustering/utils.py``.
- Tutorials updated for the renamed API and current distance-class surface:
  ``introduction_to_FGC_use_cases``, ``introduction_to_FGC_comparing_FGC_to_FI``,
  ``special_case_big_data_with_FGC``, and ``special_case_inference_with_FGC``.
- `ClusteringClara.run_clustering` now calls `distance_metric.compute_inertia`
  and `distance_metric.assign_labels` instead of the module-level helpers.
  Behavior with `DistanceRandomForestProximity` is unchanged;
  `DistanceRandomForestLCA` now produces LCA-consistent CLARA clusters
  end-to-end.
- The module-level numba functions `_calculate_inertia` / `_asign_labels` in
  `fgclustering/clustering.py` were moved to `fgclustering/distance.py`,
  renamed to `_calculate_inertia_terminals` / `_assign_labels_terminals`, and
  are dispatched from `DistanceRandomForestBase`.
- `DistanceRandomForestProximity.__init__` now accepts `min_samples_in_node` and
  validates it (must be >= 1 when not `None`).
- `DistanceRandomForestProximity.calculate_forest_encoding` now remaps the stored terminal
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
- `DistanceRandomForestProximity.calculate_forest_encoding` dispatches to a second
  ancestor-collapse branch when `max_depth_for_proximity` is configured. Behavior
  with both new parameters set to `None` (default) is byte-for-byte identical to
  previous releases.
- `DistanceRandomForestProximity` class docstring updated to enumerate both supported
  ancestor-collapse criteria and document their mutual exclusivity.
- `DistanceRandomForestProximity.__init__` now accepts `min_variance_in_node` and
  validates its value range (must be >= 0 when not ``None``). The
  ``_validate_mutually_exclusive`` call is extended to cover all three
  ancestor-collapse parameters (``min_samples_in_node``,
  ``max_depth_for_proximity``, ``min_variance_in_node``); setting any two at once
  raises ``ValueError``.
- `DistanceRandomForestProximity.calculate_forest_encoding` now performs an
  estimator-type guard and allows both ``criterion="squared_error"`` and
  ``criterion="friedman_mse"`` when ``min_variance_in_node`` is configured. A
  ``UserWarning`` is issued for ``friedman_mse`` to indicate that variance-based
  pruning is approximate in this case. Behavior with all three new parameters set
  to ``None`` (default) is byte-for-byte identical to previous releases.
- `DistanceRandomForestProximity` class docstring updated to enumerate all three
  supported ancestor-collapse criteria and document their mutual exclusivity and
  the regression-only restriction of ``min_variance_in_node``.

### Removed
- Obsolete limitation note about `DistanceRandomForestLCA` + `ClusteringClara`
  inconsistency: the underlying issue is now fixed.
- Module-level shims `_calculate_inertia` and `_asign_labels` in
  `fgclustering/clustering.py` (not part of the public API).
- Tutorial ``tutorials/inner_node_proximity_strategies.ipynb`` (superseded by
  ``special_case_tree_pruning_with_FGC.ipynb``).
- Tutorial ``tutorials/special_case_impact_of_model_complexity_on_FGC.ipynb``
  and corresponding Sphinx nblink.

### Fixed
- `plot_heatmap_classification` now uses ``color_spec["color_target_cat"]`` instead
  of ``color_spec["color_target"]`` for categorical target palettes (static and
  interactive modes).
- Error message in `DistanceRandomForestProximity.__init__` for negative
  `min_variance_in_node` values now correctly references `min_variance_in_node`
  (it had still used an outdated parameter name after earlier renames).
- CHANGELOG `[Unreleased]` consistency: stale references to outdated parameter
  names (`max_node_variance`, `min_node_variance`) replaced with
  `min_variance_in_node`.

### Added (PR-A hygiene)
- `test_min_variance_in_node_friedman_mse_emits_warning_and_proceeds` — regression
  test that `criterion="friedman_mse"` is accepted with a `UserWarning` and that
  `calculate_forest_encoding` completes normally afterwards.

### Changed (PR-A hygiene)
- `test_min_variance_in_node_rejects_absolute_error_criterion` now asserts the
  error message names both supported criteria (`squared_error`, `friedman_mse`)
  to give users an actionable hint, mirroring the existing
  `test_min_variance_in_node_rejects_classifier` style.
- Docstring of `test_min_variance_in_node_zero_matches_baseline` clarified to
  reflect the bottom-up pruning semantics (every node has variance ≥ 0, so
  nothing is pruned at θ=0).

### Added (PR-B)
- `DistanceRandomForestBase`: shared **abstract** base class (`abc.ABC`) for
  `DistanceRandomForestProximity` and `DistanceRandomForestLCA`. Holds the
  memmap configuration, the `terminals` attribute, the
  `_allocate_distance_matrix` helper, and the `remove_distance_matrix`
  cleanup logic, and declares `calculate_forest_encoding` and
  `calculate_distance_matrix` as `@abstractmethod` so that the public
  `DistanceRandomForestBase` type used in `clustering_distance_metric` /
  `distance_metric` parameters is type-safe (static type checkers see the
  full proximity interface) and direct instantiation raises `TypeError`.
  Exported from the `fgclustering` package. Both existing distance classes
  now inherit from it.
- `TestDistanceRandomForestBase`: contract tests covering the new
  ABC behavior (direct instantiation raises `TypeError`, both methods are
  marked `__isabstractmethod__`, concrete subclasses still instantiate).
- Targeted unit tests for `_compute_parent_array`, `_compute_node_depths`,
  `_build_leaf_to_ancestor_map`, and `_validate_mutually_exclusive` in a new
  `TestTreeHelpers` class.
- Per-pair invariant test `test_lca_distance_le_terminal_distance` proving LCA
  distance ≤ terminal-node distance for any pair.
- Baseline-on-regressor smoke tests for `min_samples_in_node` and
  `max_depth_for_proximity`.
- Integration test `test_forest_guided_clustering_with_lca_regressor` driving
  `forest_guided_clustering()` through `DistanceRandomForestLCA` end to end.

### Changed (PR-B)
- Type hints on `distance_metric` parameters in `ClusteringKMedoids.run_clustering`,
  `ClusteringClara.run_clustering`, `Optimizer.__init__`, and
  `forest_guided_clustering()` now reference `DistanceRandomForestBase`
  instead of `DistanceRandomForestProximity` so that `DistanceRandomForestLCA`
  is correctly advertised as supported.
- `DistanceRandomForestProximity.remove_distance_matrix` and
  `DistanceRandomForestLCA.remove_distance_matrix` deleted; both classes now
  inherit the implementation from `DistanceRandomForestBase`.
- `DistanceRandomForestProximity.calculate_distance_matrix` and
  `DistanceRandomForestLCA.calculate_distance_matrix` use the inherited
  `_allocate_distance_matrix` helper, removing ~30 LOC of duplicated memmap
  bookkeeping per class.
- `test/test_distance.py` cleanup: imports moved to module level, keyword-style
  assertion arguments (`first=`, `second=`, `expr=`, `obj=`, `a=`, `v=`)
  replaced with positional form to match the rest of the test suite, and
  `_train_regression_model` consolidated into a module-level
  `_build_regression_forest` helper shared across test classes. No semantic
  test changes.

### Fixed (PR-B, test unification)
- `test/test_forest_guided_clustering.py` import error: `DistanceRandomForestProximity`
  is now imported from `fgclustering.distance` instead of
  `fgclustering.forest_guided_clustering`. The latter only re-exports
  `DistanceRandomForestBase` after the PR-B refactor, so the previous import
  raised `ImportError` at collection time.

### Changed (PR-B, test unification)
- `test/test_distance.py` and `test/test_forest_guided_clustering.py`: shared
  fixtures consolidated. `setUp` now builds the classification forest and
  three regression forests (`squared_error`, `friedman_mse`,
  `absolute_error`) once via the module-level
  `_build_classification_forest` / `_build_regression_forest` helpers, and
  every test reuses these via `self.model_clas` / `self.X_clas` /
  `self.model_reg_squared` / `self.model_reg_friedman` /
  `self.model_reg_absolute` (and `self.model_reg` in
  `test_forest_guided_clustering.py`). All inline `make_classification` /
  `make_regression` / `RandomForestRegressor` setup inside individual tests
  removed.
- `test/test_distance.py`:
  - Per-attribute renames `self.X → self.X_clas`, `self.y → self.y_clas`,
    `self.model → self.model_clas` for clarity now that classification and
    regression fixtures coexist.
  - `test_min_samples_in_node_one_matches_baseline` and
    `test_max_depth_for_proximity_large_matches_baseline` now also assert
    baseline parity on a regressor, absorbing the old standalone
    `*_baseline_on_regressor` tests.
  - Dropped the `*_monotonicity` tests for `min_samples_in_node` and
    `max_depth_for_proximity`; off-diagonal mean monotonicity is not a
    contracted invariant of bottom-up ancestor collapse and the tests were
    flaky on small synthetic forests. Direction-of-effect is still covered
    by the `*_collapses_to_root` and `*_large_matches_baseline` tests.
  - Pairwise mutual-exclusivity tests renamed to a uniform
    `test_<param_a>_mutually_exclusive_with_<param_b>` scheme and grouped
    together with `test_all_three_mutually_exclusive`.
- `test/test_forest_guided_clustering.py`:
  - Removed the duplicated `test_forest_guided_clustering_with_lca_regressor`;
    its coverage is fully subsumed by
    `test_forest_guided_clustering_with_lca_distance_regression`, which now
    uses the shared `self.model_reg` fixture.
  - Module-level `_build_classification_forest` / `_build_regression_forest`
    helpers added (regressor configurable via `criterion`, `n_estimators`,
    `max_depth`).
- `fgclustering/distance.py`: minor docstring/formatting polish in
  `DistanceRandomForestBase` (wording "consumers" → "users"; a few short
  one-liners no longer artificially line-wrapped). No behavior change.
