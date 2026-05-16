############################################
# Imports
############################################

import shutil
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict
from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

############################################
# Utility Functions
############################################


def check_input_data(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | str,
    y_pred: pd.Series | np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """
    Normalize input data into aligned pandas objects.

    If ``y`` is provided as a column name, that column is extracted from ``X`` and removed
    from the returned feature matrix. All returned objects have their index reset to ensure
    positional alignment.

    :param X: Feature matrix or table containing the target column when ``y`` is a string.
    :type X: pd.DataFrame | np.ndarray
    :param y: Target values or the name of the target column in ``X``.
    :type y: pd.Series | np.ndarray | str
    :param y_pred: Optional predicted target values aligned with ``X``.
    :type y_pred: pd.Series | np.ndarray | None

    :raises ValueError: If ``y`` is a string but ``X`` is not a DataFrame containing that
        column.
    :raises ValueError: If ``X`` and ``y`` have different numbers of rows.
    :raises ValueError: If ``y_pred`` is provided but is not aligned with ``X`` and ``y``.

    :return: Tuple containing the normalized feature matrix, target vector, and optional
        prediction vector.
    :rtype: tuple[pd.DataFrame, pd.Series, pd.Series | None]
    """
    if isinstance(y, str):
        if isinstance(X, pd.DataFrame) and y in X.columns:
            y_data = pd.Series(X[y]).reset_index(drop=True)
            X_data = pd.DataFrame(X.drop(columns=[y])).reset_index(drop=True)
        else:
            raise ValueError(
                "X must be a pandas DataFrame and y must be a column name in X when y is a string."
            )
    else:
        y_data = pd.Series(y).reset_index(drop=True)
        X_data = pd.DataFrame(X).reset_index(drop=True)

    y_pred_data = pd.Series(y_pred).reset_index(drop=True) if y_pred is not None else None

    if len(y_data) != len(X_data):
        raise ValueError("X and y must have the same number of rows.")
    if y_pred_data is not None and len(y_data) != len(y_pred_data):
        raise ValueError("X, y and y_pred must have the same number of rows.")
    return X_data, y_data, y_pred_data


def check_input_estimator(
    estimator: Any,
) -> type[RandomForestClassifier] | type[RandomForestRegressor] | None:
    """
    Validate that an estimator is a supported Random Forest model.

    Supported estimators are instances or subclasses of
    ``RandomForestClassifier`` and ``RandomForestRegressor``.

    :param estimator: Estimator instance to validate.
    :type estimator: Any

    :return: Concrete estimator class if supported, otherwise ``None``.
    :rtype: type[RandomForestClassifier] | type[RandomForestRegressor] | None
    """
    if isinstance(estimator, RandomForestClassifier):
        return type(estimator)
    if isinstance(estimator, RandomForestRegressor):
        return type(estimator)
    return None


def matplotlib_to_plotly(
    cmap_name: str,
    pl_entries: int = 255,
) -> list:
    """
    Convert a Matplotlib colormap to a Plotly colorscale.

    The colormap is sampled uniformly over ``[0, 1]`` and converted to Plotly's
    ``[[position, color], ...]`` colorscale format using hexadecimal RGB values.

    :param cmap_name: Name of a registered Matplotlib colormap.
    :type cmap_name: str
    :param pl_entries: Number of evenly spaced samples drawn from the colormap.
    :type pl_entries: int

    :raises ValueError: If ``pl_entries < 2``.

    :return: Plotly colorscale specification.
    :rtype: list
    """
    if pl_entries < 2:
        raise ValueError(f"pl_entries must be >= 2, got {pl_entries}")
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    h = np.linspace(0, 1, pl_entries)
    colors = cmap(h)[:, :3]
    colors = [matplotlib.colors.rgb2hex(color) for color in colors]
    colorscale = [[i / (pl_entries - 1), color] for i, color in enumerate(colors)]
    return colorscale


def save_figure(
    filename_base: str,
    filename_extra: str = "",
) -> None:
    """
    Save the current Matplotlib figure to disk.

    The output filename is constructed as
    ``{parent}/{stem}{filename_extra}{suffix}``. Parent directories are created
    automatically if needed. Figures are saved with ``bbox_inches="tight"`` and
    ``dpi=300``.

    :param filename_base: Output file path including extension.
    :type filename_base: str
    :param filename_extra: Optional string inserted between the filename stem and suffix.
    :type filename_extra: str

    :return: ``None``.
    :rtype: None
    """
    p = Path(filename_base)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        p.parent / f"{p.stem}{filename_extra}{p.suffix}",
        bbox_inches="tight",
        dpi=300,
    )


def check_disk_space(
    path: str,
    required_bytes: int,
) -> bool:
    """
    Check whether sufficient free disk space is available.

    The filesystem containing ``path`` is queried with ``shutil.disk_usage`` and the
    available free space is compared against ``required_bytes``.

    :param path: Path located on the target filesystem.
    :type path: str
    :param required_bytes: Minimum required free space in bytes.
    :type required_bytes: int

    :return: ``True`` if the available free space exceeds ``required_bytes``, otherwise
        ``False``.
    :rtype: bool
    """
    total, used, free = shutil.disk_usage(path)
    return free > required_bytes


def map_clusters_to_samples(
    labels: np.ndarray,
    samples_mapping: np.ndarray | None = None,
) -> dict:
    """
    Build a mapping from cluster labels to sample identifiers.

    Each sample index is assigned to the corresponding cluster label. If
    ``samples_mapping`` is provided, mapped identifiers are stored instead of positional
    row indices.

    :param labels: Cluster labels for all samples.
    :type labels: np.ndarray
    :param samples_mapping: Optional mapping from row positions to external sample
        identifiers.
    :type samples_mapping: np.ndarray | None

    :return: Dictionary mapping cluster labels to sets of sample identifiers.
    :rtype: dict
    """
    index_vector = np.arange(len(labels))
    indices_clusters = defaultdict(set)

    for i, label in enumerate(labels):
        idx = samples_mapping[i] if samples_mapping is not None else index_vector[i]
        indices_clusters[label].add(idx)

    return dict(indices_clusters)


def check_k_range(
    k: int | tuple[int, int] | None,
) -> tuple[int, int]:
    """
    Normalize a cluster-number specification into a ``(min_k, max_k)`` tuple.

    If ``k`` is ``None``, the default range ``(2, 6)`` is returned. If ``k`` is a single
    integer, a fixed range ``(k, k)`` is returned.

    :param k: Cluster specification as ``None``, an integer, or a two-element range.
    :type k: int | tuple[int, int] | None

    :raises ValueError: If ``k`` is an integer smaller than ``2``.
    :raises ValueError: If ``k`` is not ``None``, an integer, or a valid two-element range.

    :return: Inclusive cluster-number range.
    :rtype: tuple[int, int]
    """
    if k is None:
        k_range = (2, 6)
    elif isinstance(k, int):
        if k < 2:
            raise ValueError("k must be >= 2.")
        k_range = (k, k)
    elif isinstance(k, (tuple, list)) and len(k) == 2:
        k_range = tuple(k)
    else:
        raise ValueError("k must be int, tuple of (min, max), or None.")

    return k_range


def check_sub_sample_size(
    sub_sample_size: int | float | None,
    n_samples: int,
    application: str,
    verbose: int,
) -> int:
    """
    Resolve and validate a subsample size specification.

    If ``sub_sample_size`` is ``None``, an adaptive fraction is selected based on the total
    number of samples. Float values are interpreted as fractions of ``n_samples`` and
    integer values as absolute sample counts. The final value is capped at ``n_samples``.

    :param sub_sample_size: Subsample specification as ``None``, a fraction in ``(0, 1]``,
        or a positive integer.
    :type sub_sample_size: int | float | None
    :param n_samples: Total number of available samples.
    :type n_samples: int
    :param application: Name of the calling application used in verbose messages.
    :type application: str
    :param verbose: Verbosity level controlling informational output.
    :type verbose: int

    :raises ValueError: If a float sample size is outside ``(0, 1]``.
    :raises ValueError: If an integer sample size is not positive.
    :raises TypeError: If ``sub_sample_size`` has an unsupported type.

    :return: Validated subsample size as an integer count.
    :rtype: int
    """
    if sub_sample_size is None:
        sub_sample_size = min(0.8, max(0.1, 1000 / n_samples))
        if verbose:
            print(f"Using a sample size of {sub_sample_size*100:.2f}% of the input data for {application}.")

    if isinstance(sub_sample_size, float):
        if not (0 < sub_sample_size <= 1):
            raise ValueError("If sample size is a float, it must be in (0, 1].")

        sub_sample_size = int(n_samples * sub_sample_size)

    if isinstance(sub_sample_size, int):
        if sub_sample_size == 0:
            raise ValueError("Integer sample size must be > 0.")
        sub_sample_size = min(sub_sample_size, n_samples)
    else:
        raise TypeError("Sample size must be None, float in (0, 1], or int")

    return sub_sample_size


def custom_round(x: float) -> int:
    """
    Round a float using asymmetric tie handling.

    Values with fractional part strictly greater than ``0.5`` are rounded upward using
    ``ceil``. Values with fractional part less than or equal to ``0.5`` are rounded
    downward using ``floor``.

    :param x: Value to round.
    :type x: float

    :return: Rounded integer value.
    :rtype: int
    """
    decimal = x - int(x)
    if decimal > 0.5:
        return int(np.ceil(x))
    else:
        return int(np.floor(x))
