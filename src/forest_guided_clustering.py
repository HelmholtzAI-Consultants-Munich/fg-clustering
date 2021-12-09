############################################
# imports
############################################

import numpy as np

import src.plotting as pl
import src.optimizer as opt

############################################
# Forest-guided Clustering
############################################


def forest_guided_clustering(output, data, target_column, model,   
                             max_K = 6, number_of_clusters = None, max_iter_clustering = 500,  
                             bootstraps_JI = 300, discart_value_JI = 0.6,
                             bootstraps_p_value = 10000, thr_pvalue = 0.05, random_state = 42):

    """
    This function runs forest-guided clustering algirthm for Random Forest Classifier or Regressor. 
    It computes the optimal number of clusters for a k-medoids clustering based on the distance matrix coputed from the Random Forest proximity matrix. 
    The results are plottet in a heatmap, but only features that are significantly different (based on an ANOVA test) across clusters are shown.
    Parameters
    ----------
        output: string
            Filename for heatmap plot.
        model: sklearn object
            Trained Random Forest model.
        data: DataFrame
            Input data with feature matrix. If target_column is a string it has to be a column in the
            data.
        target_column: string or numpy array
            Name of target column or target values as numpy array.
        max_K: int
            Maximum number of clusters for cluster score computation.
        thr_pvalue: int
            P-value threshold for feature filtering.
        bootstraps: int
            Number of bootstraps to compute the Jaccard Index.
        max_iter_clustering: int
            Number of iterations for k-medoids clustering.
        discart_value: float
            Minimum Jaccard Index for cluster stability.
        number_of_clusters: int
            Number of clusters for the k-medoids clustering. Leave None if number of clusters should be optimized.
        random_state: int
            Seed number for random state.

    Returns
    -------
        --- None ---
    """
    
    # check if random forest is regressor or classifier
    is_regressor = 'RandomForestRegressor' in str(type(model))
    is_classifier = 'RandomForestClassifier' in str(type(model))
    
    if is_regressor is True:
        method = "regression"
        print("Interpreting RandomForestRegressor")
    elif is_classifier is True:
        method = "classifier"
        print("Interpreting RandomForestClassifier")
    else:
        raise ValueError(f'Do not recognize {str(type(model))}. Can only work with sklearn RandomForestRegressor or RandomForestClassifier.')
    
    if type(target_column)==str:
        y = data.loc[:,target_column]
        X = data.drop(columns=[target_column])
    else:
        y = target_column
        X = data
    
    
    distanceMatrix = 1 - proximityMatrix(model, X.to_numpy())
    
    if number_of_clusters is None:
        k = opt.optimizeK(distanceMatrix, y.to_numpy(), max_K, bootstraps_JI, max_iter_clustering, discart_value_JI, method, random_state)
    else:
        k = number_of_clusters
        
    print(f"Visualizing forest guided clustering for {k} clusters")
    
    pl.plot_forest_guided_clustering(output, X, y, method, distanceMatrix, k, thr_pvalue, bootstraps_p_value, random_state)
    
    return k



def proximityMatrix(model, X, normalize=True):  
    """
    Calculate proximity matrix of Random Forest model. 
    Parameters
    ----------
        model: sklearn object
            Trained Random Forest model.
        X: DataFrame
            Feature matrix.
        normalize: Bool
            Normalize proximity matrix by number of trees in the Random Forest. 

    Returns
    -------
        proxMat: DataFrame
            Proximity matrix of Random Forest model.
    """

    terminals = model.apply(X)
    nTrees = terminals.shape[1]

    a = 0
    proxMat = 0

    for i in range(nTrees):
        a = terminals[:,i]
        proxMat += np.equal.outer(a, a)

    if normalize:
        proxMat = proxMat / nTrees

    return proxMat 