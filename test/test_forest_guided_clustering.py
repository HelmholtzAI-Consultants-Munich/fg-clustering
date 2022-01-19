############################################
# imports
############################################

import joblib
import pandas as pd

from fgclustering.forest_guided_clustering import *


############################################
# Tests
############################################

def test_forest_guided_clustering():

    data_breast_cancer = pd.read_csv('./data/data_breast_cancer.csv')
    rf = joblib.load(open('./data/random_forest_breat_cancer.joblib', 'rb'))
    
    expected_output = 2

    result = fgclustering(output='forest_guided_clustering_cancer', data=data_breast_cancer, target_column='target', model=rf,  
                                     max_K = 4, max_iter_clustering = 100, 
                                     bootstraps_JI = 30, discart_value_JI = 0.6, 
                                     bootstraps_p_value = 100, thr_pvalue = 0.001, random_state = 42)
    
    assert result == expected_output



