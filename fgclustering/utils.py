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
    X: pd.DataFrame,
    y: str | pd.Series,
    y_pred: np.ndarray | pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """
    Normalize inputs into aligned feature matrix, target vector, and optional predictions.

    If ``y`` is a column name, that column is taken as the target and removed from the
    feature matrix. Row order is reset so ``X``, ``y``, and ``y_pred`` (if given) line up
    by position.

    :param X: Feature matrix; must contain column ``y`` when ``y`` is a string.
    :type X: pd.DataFrame
    :param y: Target column name in ``X``, or target values with one entry per row of ``X``.
    :type y: str | pd.Series
    :param y_pred: Optional predictions, one per row; if omitted, the third return value is ``None``.
    :type y_pred: np.ndarray | pd.Series | None

    :return: ``(X_data, y_data, y_pred_data)`` where each is a ``DataFrame`` or ``Series`` with a fresh index, or ``y_pred_data`` is ``None``.
    :rtype: tuple[pd.DataFrame, pd.Series, pd.Series | None]
    """
    if isinstance(y, str):
        y_data = pd.Series(X[y]).reset_index(drop=True)
        X_data = pd.DataFrame(X.drop(columns=[y])).reset_index(drop=True)
    else:
        y_data = pd.Series(y).reset_index(drop=True)
        X_data = pd.DataFrame(X).reset_index(drop=True)

    y_pred_data = pd.Series(y_pred).reset_index(drop=True) if y_pred is not None else None
    return X_data, y_data, y_pred_data


def check_input_estimator(
    estimator: Any,
) -> type[RandomForestClassifier] | type[RandomForestRegressor] | None:
    """
    Return the estimator class if it is a scikit-learn random forest classifier or regressor.

    Subclasses of ``RandomForestClassifier`` or ``RandomForestRegressor`` are accepted;
    any other object yields ``None``.

    :param estimator: Fitted or unfitted estimator instance to inspect.
    :type estimator: Any

    :return: ``type(estimator)`` when ``estimator`` is a supported random forest type; otherwise ``None``.
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
    Build a Plotly ``colorscale`` from a named Matplotlib colormap.

    Samples the colormap uniformly, converts RGB stops to hex strings, and returns the
    ``[[position, color], ...]`` list Plotly expects, with positions in ``[0, 1]``.

    :param cmap_name: Registered Matplotlib colormap name (e.g. ``"viridis"``).
    :type cmap_name: str
    :param pl_entries: Number of evenly spaced samples along the colormap; must be at least 2 for a valid scale.
    :type pl_entries: int

    :return: List of ``[normalized_position, hex_color]`` pairs suitable for Plotly traces.
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
    Save the current pyplot figure to disk with an optional suffix on the basename.

    Creates parent directories as needed. The active figure is written with
    ``bbox_inches="tight"`` and ``dpi=300``. The written path is
    ``{parent}/{stem}{filename_extra}{suffix}``.

    :param filename_base: Full path including extension (e.g. ``"/out/plot.png"``).
    :type filename_base: str
    :param filename_extra: Text inserted between stem and extension (e.g. ``"_v2"`` → ``plot_v2.png``).
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
    Return whether the filesystem holding ``path`` has more free space than requested.

    Uses ``shutil.disk_usage`` on ``path`` (typically a directory on the target volume).

    :param path: Path on the device whose free space should be queried.
    :type path: str
    :param required_bytes: Minimum number of free bytes required for the check to pass.
    :type required_bytes: int

    :return: ``True`` if free space is strictly greater than ``required_bytes``, else ``False``.
    :rtype: bool
    """
    total, used, free = shutil.disk_usage(path)
    return free > required_bytes


def map_clusters_to_samples(
    labels: np.ndarray,
    samples_mapping: np.ndarray | None = None,
) -> dict:
    """
    Invert cluster labels into a mapping from each label to the set of sample indices.

    Index ``i`` in ``labels`` refers to ``samples_mapping[i]`` when ``samples_mapping`` is
    provided, otherwise to ``i``.

    :param labels: Cluster label per row, length ``n``; same length as ``samples_mapping`` when given.
    :type labels: np.ndarray
    :param samples_mapping: Optional length-``n`` array mapping row position to an external id or index.
    :type samples_mapping: np.ndarray | None

    :return: Keys are cluster labels; values are sets of (possibly remapped) sample indices in that cluster.
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
    Normalize the cluster-range argument into an inclusive ``(k_min, k_max)`` pair.

    ``None`` selects the default search window ``(2, 6)``. A single integer ``k`` requires
    ``k >= 2`` and yields ``(k, k)``. A two-element sequence is returned as a tuple of
    integers (contents are not otherwise validated).

    :param k: Fixed cluster count, ``(min, max)`` range, or ``None`` for defaults.
    :type k: int | tuple[int, int] | None

    :return: Inclusive minimum and maximum number of clusters to consider.
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
    Resolve how many rows to draw in a subsample and enforce sane bounds.

    If ``sub_sample_size`` is ``None``, the fraction is ``min(0.8, max(0.1, 1000 /
    n_samples))``; when ``verbose`` is non-zero, a message names ``application``.
    Floats are fractions in ``(0, 1]`` converted to ``int(n_samples * fraction)``.
    Integers must be positive and are capped at ``n_samples``.

    :param sub_sample_size: ``None`` (automatic), fraction in ``(0, 1]``, or positive row count.
    :type sub_sample_size: int | float | None
    :param n_samples: Total rows available in the full dataset.
    :type n_samples: int
    :param application: Label used only in optional verbose output.
    :type application: str
    :param verbose: If non-zero, print the chosen automatic fraction when ``sub_sample_size`` is ``None``.
    :type verbose: int

    :return: Integer number of rows to use, at most ``n_samples``.
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
    Round ``x`` to the nearest integer; fractional part exactly ``0.5`` rounds down.

    Uses ``ceil`` when the fractional part is greater than ``0.5``, otherwise ``floor``.

    :param x: Real value to round.
    :type x: float

    :return: Nearest integer under the rule above.
    :rtype: int
    """
    decimal = x - int(x)
    if decimal > 0.5:
        return int(np.ceil(x))
    else:
        return int(np.floor(x))
