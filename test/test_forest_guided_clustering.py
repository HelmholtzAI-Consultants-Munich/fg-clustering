############################################
# imports
############################################

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from fgclustering.forest_guided_clustering import *


############################################
# Tests
############################################

def test_forest_guided_clustering():
    
    #parameters
    max_K = 6
    method_clustering = 'pam'
    init_clustering = 'random'
    max_iter_clustering = 100
    discart_value_JI = 0.6
    bootstraps_JI = 100
    n_jobs = 3
   
    # test data
    X, y = make_classification(n_samples=300, n_features=10, n_informative=4, n_redundant=2, n_classes=2, n_clusters_per_class=1, random_state=1)
    X = pd.DataFrame(X)

    model = RandomForestClassifier(max_depth=10, max_features='sqrt', max_samples=0.8, bootstrap=True, oob_score=True, random_state=42)
    model.fit(X, y)

    data_classification = X
    data_classification['target'] = y

    # initialize and run fgclustering object
    fgc = FgClustering(model=model, data=data_classification, target_column='target')
    fgc.run(max_K=max_K, method_clustering=method_clustering, init_clustering=init_clustering, max_iter_clustering=max_iter_clustering, discart_value_JI=discart_value_JI, bootstraps_JI=bootstraps_JI, n_jobs=n_jobs)

    # obtain optimal number of clusters
    assert fgc.k == 2, "error: wrong optimal k calculated"
