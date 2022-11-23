############################################
# imports
############################################


import numpy as np
import pandas as pd
from fgclustering.utils import *
from fgclustering.statistics import compute_balanced_average_impurity, compute_total_within_cluster_variation, calculate_global_feature_importance, calculate_local_feature_importance


############################################
# Tests
############################################

def test_compute_balanced_average_impurity():

    #test data
    categorical_values = pd.Series([0,0,0,0,0,1,1,1,1,1])
    cluster_labels = np.array([0,0,0,0,0,1,1,1,1,1])
    
    result = compute_balanced_average_impurity(categorical_values, cluster_labels)
    
    assert result == 0., "error: impurity should be 0"


def test_compute_total_within_cluster_variation():

    # test data
    continuous_values = pd.Series([0,0,0,0,0,1,1,1,1,1])
    cluster_labels = np.array([0,0,0,0,0,1,1,1,1,1])
    
    result = compute_total_within_cluster_variation(continuous_values, cluster_labels)
    
    assert result == 0., "error: within cluster variation should be 0"


def test_calculate_global_feature_importance():
    
    # test if anova test filters out features 1 and 2 which are the same in both clusters and 
    # leaves features 3 and 4 which are clearly different in both clusters
    #parameters
    model_type = 'classifier'

    #test data
    X = pd.DataFrame.from_dict({'col_1': [1,1,1,1,1,0.9], 'col_2': [1,1,1,1,0.9,0.5], 'col_3': [1,1,1,0,0,1], 'col_4': [0,0,0,1,1,1]})
    y = pd.Series([0,0,0,0,0,0])
    cluster_labels = np.array([0,0,0,1,1,1])
    
    X_ranked, p_value_of_features = calculate_global_feature_importance(X, y, cluster_labels, model_type)
    
    X_ranked.drop('cluster', axis  = 1, inplace=True)
    assert list(X_ranked.columns) == ['target', 'col_4', 'col_3', 'col_2', 'col_1'], "error: global feature importance returns wrong ordering"


def test_calculate_local_feature_importance():

    # test if clusterwise importance is high for feature 2 and 4 and low for feature 1 and 3
    #parameters
    thr_pvalue = 1
    bootstraps = 100
    model_type = 'classifier'

    #test data
    X = pd.DataFrame.from_dict({'col_1': [0.9,1,0.9,1,1,0.9,1,0.9,1,0.9,1,0.9], 'col_2': [0.1,0.1,0.1,0.1,0.9,0.9,0.9,0.9,1,1,1,1], 'col_3': [0.1,0,0.1,0,0.1,0,0.1,0,0.1,0.1,0,0], 'col_4': [0,0,0,0,1,1,1,1,2,2,2,2]})
    y = pd.Series([0,0,0,0,0,0,0,0,0,0,0,1])
    cluster_labels = np.array([0,0,0,0,1,1,1,1,2,2,2,2])

    X_ranked, p_value_of_features = calculate_global_feature_importance(X, y, cluster_labels, model_type)
    for column in X.columns:
        if p_value_of_features[column] > thr_pvalue:
            X.drop(column, axis=1, inplace=True)    

    p_value_of_features_per_cluster = calculate_local_feature_importance(X_ranked, bootstraps)
    importance = 1-p_value_of_features_per_cluster
    result = importance.transpose().median()

    assert sum(result > 0.9) == 2, "error: wrong number of features with highest feature importance"



