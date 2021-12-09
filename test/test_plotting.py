from fgclustering.plotting import _scale_standard, _scale_minmax, _anova_test
import pandas as pd
import joblib
import numpy as np


def test_anova_test():
    
    # test if anova test filters out features 1 and 2 which are the same in both clusters and leaves features 3 and 4 which
    # are clearly different in both clusters
    
    data = {'col_1': [1,0.9,1,1,1,1], 'col_2': [1,1,1,1,0.9,1], 'col_3': [1,1,1,0,0,0], 'col_4': [0,0,0,1,1,1]}
    X = pd.DataFrame.from_dict(data)
    y = pd.Series([0,0,0,0,0,0])
    cluster_labels = np.array([0,0,0,1,1,1])
    thr_pvalue = 0.05
    
    result = _anova_test(X, y, cluster_labels, thr_pvalue)
    
    
    assert set(result.columns) == set(['col_3', 'col_4', 'cluster', 'target'])
    