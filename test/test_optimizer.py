############################################
# imports
############################################

import joblib
import numpy as np
import pandas as pd

from sklearn_extra.cluster import KMedoids

from fgclustering.utils import proximityMatrix
from fgclustering.optimizer import optimizeK, _compute_stability_indices, _translate_cluster_labels_to_dictionary_of_index_sets_per_cluster


############################################
# Tests
############################################

def test_optimizeK():
    
    bootstraps = 30
    max_iter_clustering = 100
    discart_value = 0.6
    random_state = 42
    n_jobs = 3

    ### test classification
    max_K = 4
    method = "classifier"

    data_breast_cancer = pd.read_csv('./data/data_breast_cancer.csv')
    model = joblib.load(open('./data/random_forest_breat_cancer.joblib', 'rb'))
    X = data_breast_cancer.drop(columns=['target']).to_numpy()
    y = data_breast_cancer.loc[:,'target'].to_numpy()
    
    distance_matrix = 1 - proximityMatrix(model, X)
    result = optimizeK(distance_matrix, y, max_K, bootstraps, max_iter_clustering, discart_value, method, random_state, n_jobs)
    
    assert result == 2, "Error optimal number of Clusters for breast cancer test case is not equal 2"

    ### test regression
    max_K = 7
    method = "regression"

    data_boston = pd.read_csv('./data/data_boston.csv')
    model = joblib.load(open('./data/random_forest_boston.joblib', 'rb'))
    X = data_boston.drop(columns=['target']).to_numpy()
    y = data_boston.loc[:,'target'].to_numpy()
    
    distance_matrix = 1 - proximityMatrix(model, X)
    result = optimizeK(distance_matrix, y, max_K, bootstraps, max_iter_clustering, discart_value, method, random_state, n_jobs)

    assert result == 5 or result == 6, "Error optimal number of Clusters for boston test case is not equal 5 or 6" 


def test_compute_stability_indices():
    
    distance_matrix = np.kron(np.eye(3,dtype=int),np.ones([10,10]))
    bootstraps = 10
    max_iter_clustering = 100
    init_clustering = 'k-medoids++'
    method_clustering = 'pam'
    random_state = 42
    n_jobs = 3
    
    
    #test 1: test if 3 different clusters are found and have maximal stability
    cluster_method = lambda X: KMedoids(n_clusters=3, random_state=random_state, init=init_clustering, method=method_clustering, max_iter=max_iter_clustering).fit(X).labels_
    labels = cluster_method(distance_matrix)
    result = _compute_stability_indices_parallel(distance_matrix, labels, cluster_method, bootstraps, n_jobs)

    assert result[0] == 1., "Clusters that should be stable are found to be unstable"
    assert result[1] == 1., "Clusters that should be stable are found to be unstable"
    assert result[2] == 1., "Clusters that should be stable are found to be unstable"
    

    #test 2: test if 2 different clusters are found and have maximal stability
    cluster_method = lambda X: KMedoids(n_clusters=2, random_state=random_state, init=init_clustering, method=method_clustering, max_iter=max_iter_clustering).fit(X).labels_
    labels = cluster_method(distance_matrix)
    result = _compute_stability_indices_parallel(distance_matrix, labels, cluster_method, bootstraps, n_jobs)
    
    assert min(result[0], result[1]) < 1., "Clusters that should be unstable are found to be stable"


def test_translate_cluster_labels_to_dictionary_of_index_sets_per_cluster():
    
    labels = [1,2,3,2,1]
    expected_output = {1: set([0,4]), 2: set([1,3]), 3: set([2])}
    
    output = _translate_cluster_labels_to_dictionary_of_index_sets_per_cluster(labels, mapping = False)

    assert output == expected_output
    
    expected_output = {1: set([10,14]), 2: set([11,13]), 3: set([12])}
    output = _translate_cluster_labels_to_dictionary_of_index_sets_per_cluster(labels, mapping = {0:10,1:11,2:12,3:13,4:14,5:15})
    
    assert output == expected_output
    
    
