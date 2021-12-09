############################################
# imports
############################################

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
