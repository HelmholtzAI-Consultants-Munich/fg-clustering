############################################
# imports
############################################

import numpy as np
import pandas as pd

import kmedoids
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from fgclustering.utils import proximityMatrix
from fgclustering.optimizer import optimizeK, _compute_stability_indices_parallel, _translate_cluster_labels_to_dictionary_of_index_sets_per_cluster


############################################
# Tests
############################################

def test_optimizeK():
    
    #parameters
    method_clustering = 'pam'
    init_clustering = 'random'
    max_iter_clustering = 100
    discart_value_JI = 0.6
    bootstraps_JI = 100
    random_state = 42
    n_jobs = 3

    ### test classification
    max_K = 5
    model_type = "classifier"

    #test data
    X, y = make_classification(n_samples=300, n_features=10, n_informative=4, n_redundant=2, n_classes=2, n_clusters_per_class=1, random_state=1)
    X = pd.DataFrame(X)

    model = RandomForestClassifier(max_depth=10, max_features='sqrt', max_samples=0.8, bootstrap=True, oob_score=True, random_state=42)
    model.fit(X, y)

    terminals = model.apply(X)
    distance_matrix = 1 - proximityMatrix(terminals)
    k = optimizeK(distance_matrix, y, model_type, max_K, method_clustering, init_clustering, max_iter_clustering, discart_value_JI, bootstraps_JI, random_state, n_jobs)

    assert k == 2, "Error optimal number of clusters for classification problem is not equal to 2"

    ### test regression
    max_K = 7
    model_type = "regression"
    discart_value_JI = 0.7

    #test data
    X, y = make_regression(n_samples=500, n_features=10, n_informative=4, n_targets=1, noise=0, random_state=1)
    X = pd.DataFrame(X)

    model = RandomForestRegressor(max_depth=5, max_features='sqrt', max_samples=0.8, bootstrap=True, oob_score=True, random_state=42)
    model.fit(X, y)

    terminals = model.apply(X)
    distance_matrix = 1 - proximityMatrix(terminals)
    k = optimizeK(distance_matrix, y, model_type, max_K, method_clustering, init_clustering, max_iter_clustering, discart_value_JI, bootstraps_JI, random_state, n_jobs)
    
    assert k == 2, "Error optimal number of clusters for regression problem is not equal to 2" 


def test_compute_stability_indices():
    
    #parameters
    method_clustering = 'pam'
    init_clustering = 'random'
    max_iter_clustering = 100
    bootstraps_JI = 100
    random_state = 42
    n_jobs = 3
    
    #test data
    distance_matrix = 1 - np.kron(np.eye(3,dtype=int),np.ones([10,10]))
    
    #test 1: test if 3 different clusters are found and have maximal stability
    cluster_method = lambda X: kmedoids.KMedoids(n_clusters=3, method = method_clustering, init=init_clustering, metric='precomputed', max_iter=max_iter_clustering, random_state=random_state).fit(X).labels_
    labels = cluster_method(distance_matrix)
    print(np.unique(labels))
    result = _compute_stability_indices_parallel(distance_matrix, labels, cluster_method, bootstraps_JI, n_jobs)
    assert result[0] == 1., "Clusters that should be stable are found to be unstable"
    assert result[1] == 1., "Clusters that should be stable are found to be unstable"
    assert result[2] == 1., "Clusters that should be stable are found to be unstable"
    
    #test 2: test if 2 different clusters are found that don't have maximal stability
    cluster_method = lambda X: kmedoids.KMedoids(n_clusters=2, method = method_clustering, init=init_clustering, metric='precomputed', max_iter=max_iter_clustering, random_state=random_state).fit(X).labels_
    labels = cluster_method(distance_matrix)
    print(np.unique(labels))
    result = _compute_stability_indices_parallel(distance_matrix, labels, cluster_method, bootstraps_JI, n_jobs)
    
    assert min(result[0], result[1]) < 1., "Clusters that should be unstable are found to be stable"


def test_translate_cluster_labels_to_dictionary_of_index_sets_per_cluster():
    
    #test data
    labels = [1,2,3,2,1]
    mapping = {0:10,1:11,2:12,3:13,4:14,5:15}
    
    # test translation without mapping
    dictionary_of_index_sets_per_cluster = _translate_cluster_labels_to_dictionary_of_index_sets_per_cluster(labels, mapping = False)
    assert dictionary_of_index_sets_per_cluster == {1: set([0,4]), 2: set([1,3]), 3: set([2])}, "error: wrong dictionary of index sets"
    
    # test translation with mapping
    dictionary_of_index_sets_per_cluster = _translate_cluster_labels_to_dictionary_of_index_sets_per_cluster(labels, mapping = mapping)
    assert dictionary_of_index_sets_per_cluster == {1: set([10,14]), 2: set([11,13]), 3: set([12])}, "error: wrong dictionary of index sets"
