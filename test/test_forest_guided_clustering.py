############################################
# imports
############################################

import joblib
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname('/Users/helena.pelin/Desktop/Workmap/Vouchers/FGClustering/Code/fg-clustering/'))
from fgclustering.forest_guided_clustering import *


############################################
# Tests
############################################

def test_forest_guided_clustering():

    data_breast_cancer = pd.read_csv('./data/data_breast_cancer.csv')
    rf = joblib.load(open('./data/random_forest_breat_cancer.joblib', 'rb'))
    
    expected_output = 2
    
    # initialize and run fgclustering object
    fgc = FgClustering(model=rf, data=data_breast_cancer, target_column='target')
    fgc.run(max_K = 4, bootstraps_JI = 30, max_iter_clustering = 100, discart_value_JI = 0.6)

    # obtain optimal number of clusters and vector that contains the cluster label of each data point
    result = fgc.k
    
    assert result == expected_output


# def test_helena(number_of_clusters = None):
#     #data = pd.read_csv('./data/data_iris.csv')
#     data = pd.read_csv('./data/data_iris_new.csv')
#     rf = joblib.load(open('./data/random_forest_iris.joblib', 'rb'))
#
#     # initialize and run fgclustering object
#     print(data.shape)
#     fgc = FgClustering(model=rf, data=data, target_column='target')
#     fgc.run(number_of_clusters = number_of_clusters, max_iter_clustering = 1000, n_jobs=3)
#
#     print(len(fgc.cluster_labels))
#     print(fgc.p_value_of_features)
#
#
# test_helena()
