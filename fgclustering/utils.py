############################################
# imports
############################################

import matplotlib.colors
import numpy as np
from numba import njit, prange

import matplotlib

############################################
# functions
############################################


def log_transform(p_values: list, epsilon: float = 1e-50):
    """
    Apply a log transformation to p-values to enhance numerical stability and highlight differences.
    Adds a small constant `epsilon` to avoid taking the log of zero and normalizes by dividing by
    the log of `epsilon`.

    :param p_values: List of p-values to be transformed.
    :type p_values: list
    :param epsilon: Small constant added to p-values to avoid log of zero. Defaults to 1e-50.
    :type epsilon: float, optional
    :return: Transformed p-values after log transformation.
    :rtype: numpy.ndarray
    """
    # add a small constant epsilon
    p_values = np.clip(p_values, epsilon, 1)
    return -np.log(p_values) / -np.log(epsilon)


@njit
def _calculate_proximityMatrix(terminals, normalize):
    """Calculate proximity matrix given leaf indices from the random forest model.
    Function is paralellized with numba and especially useful in case of big datasets or forests with a large number of estimators

    :param terminals: ndarray of shape (n_samples, n_estimators), result of apply() method of RandomForest; for each tree in the forest, it contais leaf indices a sample ended up in.
    :type terminals: numpy array
    :param normalize: Normalize proximity matrix by number of trees in the Random Forest, defaults to True.
    :type normalize: bool, optional
    :return: calculated proximity matrix of Random Forest model
    :rtype: numpy array
    """

    n = terminals.shape[0]
    proxMat = np.zeros((n, n))
    for i in prange(n):
        for j in prange(i, n):
            proxMat[i, j] = np.sum(terminals[i, :] == terminals[j, :])
    proxMat = proxMat + proxMat.T - np.eye(n) * proxMat[0, 0]

    if normalize:
        proxMat = proxMat / terminals.shape[1]

    return proxMat


def proximityMatrix(model, X, normalize=True):
    """Calculate proximity matrix of Random Forest model.

    :param model: Trained Random Forest model.
    :type model: sklearn.ensemble
    :param X: Feature matrix.
    :type X: pandas.DataFrame
    :param normalize: Normalize proximity matrix by number of trees in the Random Forest, defaults to True.
    :type normalize: bool, optional
    :return: Proximity matrix of Random Forest model.
    :rtype: numpy array
    """

    terminals = model.apply(X)

    return _calculate_proximityMatrix(terminals, normalize)


def matplotlib_to_plotly(cmap_name: str, pl_entries: int = 255):
    """
    Converts a Matplotlib colormap to a Plotly colorscale.

    :param cmap_name: Name of the Matplotlib colormap.
    :type cmap_name: str
    :param pl_entries: Number of color entries in the Plotly colorscale.
    :type pl_entries: int
    :return: A Plotly-compatible colorscale
    :rtype: list
    """
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    h = np.linspace(0, 1, pl_entries)
    colors = cmap(h)[:, :3]
    colors = [matplotlib.colors.rgb2hex(color) for color in colors]
    colorscale = [[i / (pl_entries - 1), color] for i, color in enumerate(colors)]
    return colorscale
