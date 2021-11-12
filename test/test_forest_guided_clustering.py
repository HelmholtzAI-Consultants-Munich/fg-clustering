import pandas as pd
import joblib

from src.forest_guided_clustering import *


def test_forest_guided_clustering():
    #input
    data_breast_cancer = pd.read_csv('./data/data_breast_cancer.csv')
    rf = joblib.load(open('./data/random_forest_breat_cancer.joblib', 'rb'))
    
    expected_output = 2
    
    result = forest_guided_clustering(output='forest_guided_clustering_classification', model=rf, data=data_breast_cancer, target_column='target', max_K = 4, thr_pvalue = 0.001, 
                                         bootstraps = 30, max_iter_clustering = 100, discart_value = 0.6, number_of_clusters = None, random_state = 42)
    
    assert result == expected_output



def test_proximityMatrix():  
    
    data_breast_cancer = pd.read_csv('./data/data_breast_cancer.csv')
    model = joblib.load(open('./data/random_forest_breat_cancer.joblib', 'rb'))
    X = data_breast_cancer.drop(columns=['target']).to_numpy()
    
    result = proximityMatrix(model, X)
    
    dim1, dim2 = result.shape
    
    assert dim1 == dim2, "error proximity matrix not quadratic"
    assert dim1 == len(X), "error proximity matrix has wrong dimensions"
    
    assert np.diag(result).min() == 1.
    assert np.diag(result).max() == 1.