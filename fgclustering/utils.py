############################################
# Imports
############################################

import shutil
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.colors

from collections import defaultdict
from typing import Union, Tuple, Any, Optional

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

############################################
# Utility Functions
############################################


def check_input_data(
    X: pd.DataFrame,
    y: Union[str, pd.Series],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits the input into features and target. If `y` is a string, it's interpreted as the name of the target column in `X`.

    :param X: Input features as a DataFrame or array-like object.
    :type X: pandas.DataFrame
    :param y: Target values or name of the target column.
    :type y: Union[str, pandas.Series]

    :return: Tuple of feature DataFrame and target Series.
    :rtype: Tuple[pd.DataFrame, pd.Series]
    """
    if isinstance(y, str):
        y_data = pd.Series(X[y]).reset_index(drop=True)
        X_data = pd.DataFrame(X.drop(columns=[y])).reset_index(drop=True)
    else:
        y_data = pd.Series(y).reset_index(drop=True)
        X_data = pd.DataFrame(X).reset_index(drop=True)

    return X_data, y_data


def check_input_estimator(
    estimator: Any,
) -> Tuple[bool, str]:
    """
    Checks whether the given estimator is a supported RandomForest model and determines its type.

    :param estimator: Trained model to validate.
    :type estimator: Any

    :return: Tuple indicating whether the estimator is valid and its model type ('cla' or 'reg').
    :rtype: Tuple[bool, str]
    """
    valid_estimator = False
    model_type = "invalid"
    if isinstance(estimator, RandomForestClassifier):
        valid_estimator = True
        model_type = "cla"
    elif isinstance(estimator, RandomForestRegressor):
        valid_estimator = True
        model_type = "reg"

    return valid_estimator, model_type


def matplotlib_to_plotly(
    cmap_name: str,
    pl_entries: Optional[int] = 255,
) -> list:
    """
    Converts a matplotlib colormap to a Plotly-compatible colorscale.

    :param cmap_name: Name of the matplotlib colormap to convert.
    :type cmap_name: str
    :param pl_entries: Number of color entries to generate.
    :type pl_entries: Optional[int]

    :return: List of Plotly-compatible color mappings.
    :rtype: list
    """
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    h = np.linspace(0, 1, pl_entries)
    colors = cmap(h)[:, :3]
    colors = [matplotlib.colors.rgb2hex(color) for color in colors]
    colorscale = [[i / (pl_entries - 1), color] for i, color in enumerate(colors)]
    return colorscale


def check_disk_space(
    path: str,
    required_bytes: int,
) -> bool:
    """
    Checks whether the specified directory has sufficient free disk space.

    :param path: Directory to check for available space.
    :type path: str
    :param required_bytes: Number of bytes required.
    :type required_bytes: int

    :return: True if sufficient space is available, False otherwise.
    :rtype: bool
    """
    total, used, free = shutil.disk_usage(path)
    return free > required_bytes


def map_clusters_to_samples(
    labels: np.ndarray,
    samples_mapping: Optional[np.ndarray] = None,
) -> dict:
    """
    Maps sample indices to their corresponding cluster labels.

    :param labels: Cluster label for each sample.
    :type labels: np.ndarray
    :param samples_mapping: Optional mapping of internal to external indices.
    :type samples_mapping: Optional[np.ndarray]

    :return: Dictionary mapping cluster labels to sets of sample indices.
    :rtype: dict
    """
    index_vector = np.arange(len(labels))
    indices_clusters = defaultdict(set)

    for i, label in enumerate(labels):
        idx = samples_mapping[i] if samples_mapping is not None else index_vector[i]
        indices_clusters[label].add(idx)

    return dict(indices_clusters)


def check_k_range(
    k: Union[int, Tuple[int, int], None],
) -> Tuple[int, int]:
    """
    Validates and returns a standardized k range for clustering.

    :param k: Number of clusters or range of cluster values.
    :type k: Union[int, Tuple[int, int], None]

    :return: Tuple representing the range of k values.
    :rtype: Tuple[int, int]
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
    sub_sample_size: Union[int, float, None],
    n_samples: int,
    application: str,
    verbose: int,
) -> int:
    """
    Validates and computes the number of samples to use in a subsample.

    :param sub_sample_size: Fraction (float), fixed count (int), or None for auto.
    :type sub_sample_size: float, int, or None
    :param n_samples: Total number of samples available.
    :type n_samples: int
    :param application: Name of the application for logging purposes.
    :type application: str
    :param verbose: Verbosity level (0 = silent, 1 = progress messages).
    :type verbose: int

    :return: Validated and resolved integer subsample size.
    :rtype: int
    """
    if sub_sample_size is None:
        sub_sample_size = min(0.8, max(0.1, 1000 / n_samples))
        if verbose:
            print(f"Using a sample size of {sub_sample_size*100} % of the input data for {application}.")

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
    decimal = x - int(x)
    if decimal > 0.5:
        return int(np.ceil(x))
    else:
        return int(np.floor(x))
