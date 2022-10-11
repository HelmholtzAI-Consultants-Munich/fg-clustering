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
    
    expected_output = 2
    
    # initialize and run fgclustering object
    fgc = FgClustering(model=rf, data=data_breast_cancer, target_column='target')
    fgc.run(max_K = 6, bootstraps_JI = 30, max_iter_clustering = 500, discart_value_JI = 0.6, n_jobs = 2)

    # obtain optimal number of clusters and vector that contains the cluster label of each data point
    result = fgc.k

    assert result == expected_output

