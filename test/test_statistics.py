############################################
# imports
############################################

import joblib
import numpy as np
import pandas as pd

from fgclustering.statistics import compute_balanced_average_impurity, compute_total_within_cluster_variation, feature_ranking


############################################
# Tests
############################################

def test_compute_balanced_average_impurity():
    
    expected_result = 0.

    categorical_values = pd.Series([0,0,0,0,0,1,1,1,1,1])
    cluster_labels = np.array([0,0,0,0,0,1,1,1,1,1])
    
    result = compute_balanced_average_impurity(categorical_values, cluster_labels)
    
    assert expected_result == result


def test_compute_total_within_cluster_variation():
    
    expected_result = 0.

    continuous_values = pd.Series([0,0,0,0,0,1,1,1,1,1])
    cluster_labels = np.array([0,0,0,0,0,1,1,1,1,1])
    
    result = compute_total_within_cluster_variation(continuous_values, cluster_labels)
    
    assert expected_result == result


def test_feature_ranking():
    
    # test if anova test filters out features 1 and 2 which are the same in both clusters and 
    # leaves features 3 and 4 which are clearly different in both clusters
    
    X = pd.DataFrame.from_dict({'col_1': [1,0.9,1,1,1,1], 'col_2': [1,1,1,1,0.9,1], 'col_3': [1,1,1,0,0,0], 'col_4': [0,0,0,1,1,1]})
    y = pd.Series([0,0,0,0,0,0])
    cluster_labels = np.array([0,0,0,1,1,1])
    thr_pvalue = 0.05
    
    result = feature_ranking(X, y, cluster_labels, thr_pvalue)
    
    assert set(result.columns) == set(['col_3', 'col_4', 'cluster', 'target'])


def get_feature_importance_clusterwise():

    '''To Do
    '''
    assert True



