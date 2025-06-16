############################################
# Imports
############################################

import shutil
import numpy as np
import pandas as pd

from collections import defaultdict

import matplotlib
import matplotlib.colors

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

############################################
# Utility Functions
############################################


def check_input_data(X, y):
    if isinstance(y, str):
        y_data = pd.Series(X[y]).reset_index(drop=True)
        X_data = pd.DataFrame(X.drop(columns=[y])).reset_index(drop=True)
    else:
        y_data = pd.Series(y).reset_index(drop=True)
        X_data = pd.DataFrame(X).reset_index(drop=True)

    return X_data, y_data


def check_input_estimator(estimator):
    valid_estimator = False
    model_type = None
    if isinstance(estimator, RandomForestClassifier):
        valid_estimator = True
        model_type = "cla"
    elif isinstance(estimator, RandomForestRegressor):
        valid_estimator = True
        model_type = "reg"

    return valid_estimator, model_type


def matplotlib_to_plotly(cmap_name: str, pl_entries: int = 255):

    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    h = np.linspace(0, 1, pl_entries)
    colors = cmap(h)[:, :3]
    colors = [matplotlib.colors.rgb2hex(color) for color in colors]
    colorscale = [[i / (pl_entries - 1), color] for i, color in enumerate(colors)]
    return colorscale


def check_disk_space(path: str, required_bytes: int) -> bool:
    total, used, free = shutil.disk_usage(path)
    return free > required_bytes


def map_clusters_to_samples(labels, samples_mapping=None):
    index_vector = np.arange(len(labels))
    indices_clusters = defaultdict(set)

    for i, label in enumerate(labels):
        idx = samples_mapping[i] if samples_mapping is not None else index_vector[i]
        indices_clusters[label].add(idx)

    return dict(indices_clusters)


def check_k_range(k):
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


def check_sub_sample_size(sub_sample_size, n_samples, verbose=0):
    if sub_sample_size is None:
        sub_sample_size = min(0.8, max(0.1, 1000 / n_samples))
        if verbose:
            print(f"Using a sample size of {sub_sample_size*100} % of the input data.")

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
