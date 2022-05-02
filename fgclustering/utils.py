############################################
# imports
############################################

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


############################################
# functions
############################################

def scale_standard(X):
    '''Feature Scaling with StandardScaler.

    :param X: Feature matrix.
    :type X: pandas.DataFrame
    :return: Standardized feature matrix.
    :rtype: pandas.DataFrame
    '''    
    X = X.copy()
    SCALE = StandardScaler()
    SCALE.fit(X)

    X_scale = pd.DataFrame(SCALE.transform(X))
    X_scale.columns = X.columns
    X_scale.reset_index(inplace=True,drop=True)

    return X_scale


def scale_minmax(X):
    '''Feature Scaling with MinMaxScaler.

    :param X: Feature matrix.
    :type X: pandas.DataFrame
    :return: Standardized feature matrix.
    :rtype: pandas.DataFrame
    ''' 
    X = X.copy()
    SCALE = MinMaxScaler()
    SCALE.fit(X)

    X_scale = pd.DataFrame(SCALE.transform(X))
    X_scale.columns = X.columns
    X_scale.reset_index(inplace=True,drop=True)

    return X_scale


def proximityMatrix(model, X, normalize=True):  
    '''Calculate proximity matrix of Random Forest model. 

    :param model: Trained Random Forest model.
    :type model: sklearn.ensemble
    :param X: Feature matrix.
    :type X: pandas.DataFrame
    :param normalize: Normalize proximity matrix by number of trees in the Random Forest, defaults to True.
    :type normalize: bool, optional
    :return: Proximity matrix of Random Forest model.
    :rtype: pandas.DataFrame
    '''
    terminals = model.apply(X)
    nTrees = terminals.shape[1]

    a = 0
    proxMat = 0

    for i in range(nTrees):
        a = terminals[:,i]
        proxMat += np.equal.outer(a, a)

    if normalize:
        proxMat = proxMat / nTrees

    return proxMat