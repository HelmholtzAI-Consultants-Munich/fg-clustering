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
    Normalize input data into aligned feature, target, and optional prediction objects.

    If ``y`` is given as a column name, that column is extracted from ``X`` as the target
    and removed from the feature matrix. All returned objects have their index reset so
    they align by position.

    :param X: Input data containing features and, when ``y`` is a string, the target column.
    :type X: pd.DataFrame | np.ndarray
    :param y: Target column name in ``X``, or a target vector aligned with the rows of ``X``.
    :type y: pd.Series | np.ndarray | str
    :param y_pred: Optional prediction values aligned with the rows of ``X``.
    :type y_pred: pd.Series | np.ndarray | None

    :return: Tuple containing the feature matrix, target vector, and optional prediction vector.
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
        raise ValueError("X, y and y_pred must have the same number of rows.")
    if y_pred_data is not None and len(y_data) != len(y_pred_data):
        raise ValueError("X, y and y_pred must have the same number of rows.")
    return X_data, y_data, y_pred_data


def check_input_estimator(
    estimator: Any,
) -> type[RandomForestClassifier] | type[RandomForestRegressor] | None:
    """
    Check whether an estimator is a supported random forest model type.

    Accepts instances of :class:`sklearn.ensemble.RandomForestClassifier` and
    :class:`sklearn.ensemble.RandomForestRegressor`, including subclasses, and returns
    their concrete class. Any other estimator returns ``None``.

    :param estimator: Estimator instance to validate.
    :type estimator: Any

    :return: Estimator class if it is a supported random forest type, otherwise ``None``.
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
    Convert a Matplotlib colormap into a Plotly colorscale.

    The colormap is sampled uniformly over ``[0, 1]``, converted to hexadecimal RGB
    values, and returned in the ``[[position, color], ...]`` format expected by Plotly.

    :param cmap_name: Name of a registered Matplotlib colormap.
    :type cmap_name: str
    :param pl_entries: Number of evenly spaced samples taken from the colormap.
    :type pl_entries: int

    :return: Plotly colorscale as a list of normalized positions and hex colors.
    :rtype: list
    """
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
    Save the current Matplotlib figure to disk with an optional filename suffix.

    The output path is built as ``{parent}/{stem}{filename_extra}{suffix}``. Parent
    directories are created if needed, and the figure is saved with tight bounding box
    and ``dpi=300``.

    :param filename_base: Output path including file extension.
    :type filename_base: str
    :param filename_extra: Optional text inserted between the stem and suffix of the filename.
    :type filename_extra: str

    :return: ``None``
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
    Check whether the filesystem containing ``path`` has sufficient free space.

    Uses :func:`shutil.disk_usage` to query the free space on the device that holds
    ``path`` and compares it to ``required_bytes``.

    :param path: Path on the target filesystem.
    :type path: str
    :param required_bytes: Minimum number of free bytes required.
    :type required_bytes: int

    :return: ``True`` if the available free space is greater than ``required_bytes``, otherwise ``False``.
    :rtype: bool
    """
    total, used, free = shutil.disk_usage(path)
    return free > required_bytes


def map_clusters_to_samples(
    labels: np.ndarray,
    samples_mapping: np.ndarray | None = None,
) -> dict:
    """
    Map cluster labels to the corresponding sample indices.

    For each entry in ``labels``, assigns the row index to the cluster label. If
    ``samples_mapping`` is provided, the mapped sample identifier is used instead of the
    row position.

    :param labels: Cluster label assigned to each sample.
    :type labels: np.ndarray
    :param samples_mapping: Optional mapping from row positions to external sample identifiers.
    :type samples_mapping: np.ndarray | None

    :return: Dictionary mapping each cluster label to a set of sample indices or identifiers.
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
    Normalize the cluster range specification to a ``(k_min, k_max)`` tuple.

    If ``k`` is ``None``, the default range ``(2, 6)`` is returned. If ``k`` is a single
    integer, it is interpreted as a fixed number of clusters and returned as ``(k, k)``.
    If ``k`` is a two-element tuple or list, it is converted to a tuple.

    :param k: Cluster specification as a single integer, a two-element range, or ``None``.
    :type k: int | tuple[int, int] | None

    :return: Minimum and maximum number of clusters.
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
    Resolve and validate the subsample size to use for an application.

    If ``sub_sample_size`` is ``None``, an automatic fraction is chosen as
    ``min(0.8, max(0.1, 1000 / n_samples))``. Float values are interpreted as fractions
    of ``n_samples``, and integer values are interpreted as absolute sample counts. The
    returned value is always capped at ``n_samples``.

    :param sub_sample_size: Sample size as ``None``, fraction in ``(0, 1]``, or positive integer count.
    :type sub_sample_size: int | float | None
    :param n_samples: Total number of available samples.
    :type n_samples: int
    :param application: Name of the calling application, used only in verbose output.
    :type application: str
    :param verbose: If non-zero, print the automatically selected sample fraction.
    :type verbose: int

    :return: Number of samples to draw.
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
    Round a float to the nearest integer using a custom tie-breaking rule.

    Values with a fractional part greater than ``0.5`` are rounded up with ``ceil``.
    Values with a fractional part less than or equal to ``0.5`` are rounded down with
    ``floor``.

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
