############################################
# imports
############################################

import pandas as pd

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from fgclustering.utils import *


############################################
# Tests
############################################


def test_proximityMatrix():
    # test data
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=4,
        n_redundant=2,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=1,
    )
    X = pd.DataFrame(X)

    model = RandomForestClassifier(
        max_depth=10,
        max_features="sqrt",
        max_samples=0.8,
        bootstrap=True,
        oob_score=True,
        random_state=42,
    )
    model.fit(X, y)

    result = proximityMatrix(model, X)

    dim1, dim2 = result.shape

    assert dim1 == dim2, "error: proximity matrix not quadratic"
    assert dim1 == len(X), "error: proximity matrix has wrong dimensions"

    assert (
        np.diag(result).min() == 1.0
    ), "error: proximity matrix should have ones on diagonal"
    assert (
        np.diag(result).max() == 1.0
    ), "error: proximity matrix should have ones on diagonal"

    assert np.allclose(result, result.T), "error: proximity matrix should be symmetric"
