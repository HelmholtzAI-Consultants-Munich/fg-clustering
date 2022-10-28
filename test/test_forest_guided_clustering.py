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
    
    #parameters
    max_K = 6
    method_clustering = 'pam'
    init_clustering = 'k-medoids++'
    max_iter_clustering = 100
    discart_value_JI = 0.6
    bootstraps_JI = 100
    n_jobs = 3
   
    # test data
    data_breast_cancer = pd.read_csv('./data/data_breast_cancer.csv')
    rf = joblib.load(open('./data/random_forest_breat_cancer.joblib', 'rb'))

    # initialize and run fgclustering object
    fgc = FgClustering(model=rf, data=data_breast_cancer, target_column='target')
    fgc.run(max_K=max_K, method_clustering=method_clustering, init_clustering=init_clustering, max_iter_clustering=max_iter_clustering, discart_value_JI=discart_value_JI, bootstraps_JI=bootstraps_JI, n_jobs=n_jobs)

    # obtain optimal number of clusters
    assert fgc.k == 2, "error: wrong optimal k calculated"

