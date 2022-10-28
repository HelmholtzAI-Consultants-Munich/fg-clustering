############################################
# imports
############################################

import joblib
import pandas as pd


from fgclustering.utils import *


############################################
# Tests
############################################


def test_proximityMatrix():  
    
    #test data
    data_breast_cancer = pd.read_csv('./data/data_breast_cancer.csv')
    model = joblib.load(open('./data/random_forest_breat_cancer.joblib', 'rb'))
    X = data_breast_cancer.drop(columns=['target']).to_numpy()
    
    result = proximityMatrix(model, X)
    
    dim1, dim2 = result.shape
    
    assert dim1 == dim2, "error: proximity matrix not quadratic"
    assert dim1 == len(X), "error: proximity matrix has wrong dimensions"
    
    assert np.diag(result).min() == 1., "error: proximity matrix should have ones on diagonal"
    assert np.diag(result).max() == 1., "error: proximity matrix should have ones on diagonal"