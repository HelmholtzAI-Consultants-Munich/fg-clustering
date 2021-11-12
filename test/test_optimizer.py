from src.optimizer import *
import pandas as pd
import joblib
import numpy as np

from src.forest_guided_clustering import *
from src.optimizer import _translate_cluster_labels_to_dictionary_of_index_sets_per_cluster

def test_optimizeK():
    max_K = 4
    bootstraps = 30
    max_iter_clustering = 100
    discart_value = 0.6
    random_state = 42
    method = "classifier"

    data_breast_cancer = pd.read_csv('./data/data_breast_cancer.csv')
    model = joblib.load(open('./data/random_forest_breat_cancer.joblib', 'rb'))
    X = data_breast_cancer.drop(columns=['target']).to_numpy()
    y = data_breast_cancer.loc[:,'target'].to_numpy()
    
    distance_matrix = 1 - proximityMatrix(model, X)
    
    result = optimizeK(distance_matrix, y, max_K, bootstraps, max_iter_clustering, discart_value, method, random_state)
    
    assert result == 2, "Error optimal number of Clusters for breast cancer test case is not equal 2"



def test_compute_stability_indices():
    
    distance_matrix = np.kron(np.eye(3,dtype=int),np.ones([10,10]))
    bootstraps = 10
    random_state = 42
    
    #test 1: test if 3 different clusters are found and have maximal stability
    cluster_method = lambda X: KMedoids(n_clusters=3, random_state=42, init = 'build', method = "pam", max_iter=100).fit(X).labels_
    
    result = compute_stability_indices(distance_matrix, cluster_method, bootstraps, random_state)
    
    assert result[0] == 1., "Clusters that should be stable are found to be unstable"
    assert result[1] == 1., "Clusters that should be stable are found to be unstable"
    assert result[2] == 1., "Clusters that should be stable are found to be unstable"
    
    #test 2: test if 3 different clusters are found and have maximal stability
    cluster_method = lambda X: KMedoids(n_clusters=2, random_state=42, init = 'build', method = "pam", max_iter=100).fit(X).labels_
    
    result = compute_stability_indices(distance_matrix, cluster_method, bootstraps, random_state)
    
    
    
    assert min(result[0], result[1]) < 1., "Clusters that should be unstable are found to be stable"




def test_translate_cluster_labels_to_dictionary_of_index_sets_per_cluster():
    
    labels = [1,2,3,2,1]
    
    expected_output = {1: set([0,4]), 2: set([1,3]), 3: set([2])}
    
    output = _translate_cluster_labels_to_dictionary_of_index_sets_per_cluster(labels, mapping = False)
    print(output)
    assert output == expected_output
    
    expected_output = {1: set([10,14]), 2: set([11,13]), 3: set([12])}
    output = _translate_cluster_labels_to_dictionary_of_index_sets_per_cluster(labels, mapping = {0:10,1:11,2:12,3:13,4:14,5:15})
    
    assert output == expected_output
    
    
def test_compute_balanced_average_impurity():
    
    y = pd.Series([0,0,0,0,0,1,1,1,1,1])
    labels = np.array([0,0,0,0,0,1,1,1,1,1])
    expected_result = 0.
    
    result = compute_balanced_average_impurity(y, labels)
    
    assert expected_result == result
    
    
def test_compute_total_within_cluster_variation():
    
    y = pd.Series([0,0,0,0,0,1,1,1,1,1])
    labels = np.array([0,0,0,0,0,1,1,1,1,1])
    expected_result = 0.
    result = compute_total_within_cluster_variation(y, labels)
    
    assert expected_result == result