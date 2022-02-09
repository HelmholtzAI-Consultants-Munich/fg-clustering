############################################
# imports
############################################

import fgclustering.utils as utils
import fgclustering.optimizer as optimizer
import fgclustering.plotting as plotting

############################################
# Forest-guided Clustering
############################################


def fgclustering(output, data, target_column, model,   
                 max_K = 8, number_of_clusters = None, max_iter_clustering = 500,  
                 bootstraps_JI = 300, discart_value_JI = 0.6,
                 bootstraps_p_value = 1000, thr_pvalue = 0.01, random_state = 42):
    '''Run forest-guided clustering algirthm for Random Forest Classifier or Regressor. The optimal number of clusters 
    for a k-medoids clustering is computed, based on the distance matrix computed from the Random Forest proximity matrix. 
    Features are ranked and filtered based on statistical tests (ANOVA for continuous features, chi square for categorical features).
    Feature distribution per cluster is shown in a heatmap and boxplots. Feature importance is plotted to show 
    the importance of each feature for each cluster, measured by variance and impurity of the feature within the cluster, 
    i.e. the higher the feature importance, the lower the feature variance/impurity within the cluster.

    :param output: Filename to save plot.
    :type output: str
    :param data: Input data with feature matrix. 
        If target_column is a string it has to be a column in the data.
    :type data: pandas.DataFrame
    :param target_column: Name of target column or target values as numpy array.
    :type target_column: str or numpy.ndarray
    :param model: Trained Random Forest model.
    :type model: sklearn.ensemble
    :param max_K: Maximum number of clusters for cluster score computation, defaults to 6
    :type max_K: int, optional
    :param number_of_clusters: Number of clusters for the k-medoids clustering. 
        Leave None if number of clusters should be optimized, defaults to None
    :type number_of_clusters: int, optional
    :param max_iter_clustering: Number of iterations for k-medoids clustering, defaults to 500
    :type max_iter_clustering: int, optional
    :param bootstraps_JI: Number of bootstraps to compute the Jaccard Index, defaults to 300
    :type bootstraps_JI: int, optional
    :param discart_value_JI: Minimum Jaccard Index for cluster stability, defaults to 0.6
    :type discart_value_JI: float, optional
    :param bootstraps_p_value: Number of bootstraps to compute the p-value of feature importance, defaults to 10000
    :type bootstraps_p_value: int, optional
    :param thr_pvalue: P-value threshold for feature filtering, defaults to 0.05
    :type thr_pvalue: float, optional
    :param random_state: Seed number for random state, defaults to 42
    :type random_state: int, optional
    :return: Optimal number of clusters.
    :rtype: int
    '''
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
    
    distanceMatrix = 1 - utils.proximityMatrix(model, X.to_numpy())
    
    if number_of_clusters is None:
        k = optimizer.optimizeK(distanceMatrix, y.to_numpy(), max_K, bootstraps_JI, max_iter_clustering, discart_value_JI, method, random_state)
    else:
        k = number_of_clusters
        
    print(f"Visualizing forest guided clustering for {k} clusters")
    plotting.plot_forest_guided_clustering(output, X, y, method, distanceMatrix, k, thr_pvalue, bootstraps_p_value, random_state)
    
    return k
