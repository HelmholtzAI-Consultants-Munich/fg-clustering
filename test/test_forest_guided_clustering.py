############################################
# imports
############################################

import joblib
import pandas as pd
import sys
import os
from fgclustering.forest_guided_clustering import *
import matplotlib.pyplot as plt


############################################
# Tests
############################################

def test_forest_guided_clustering():
    
    data_breast_cancer = pd.read_csv('./data/data_breast_cancer.csv')
    rf = joblib.load(open('./data/random_forest_breat_cancer.joblib', 'rb'))
    
    max_K = 6
    bootstraps_JI = 100
    max_iter_clustering = 100
    init_clustering = 'k-medoids++'
    method_clustering = 'pam'
    discart_value_JI = 0.6
    n_jobs = 3
    
    # initialize and run fgclustering object
    fgc = FgClustering(model=rf, data=data_breast_cancer, target_column='target')
    fgc.run(max_K=max_K, bootstraps_JI=bootstraps_JI, max_iter_clustering=max_iter_clustering, init_clustering=init_clustering, method_clustering=method_clustering, discart_value_JI=discart_value_JI, n_jobs=n_jobs)

    # obtain optimal number of clusters and vector that contains the cluster label of each data point
    result = fgc.k

    assert result == 2, "error: wrong optimal k calculated"

