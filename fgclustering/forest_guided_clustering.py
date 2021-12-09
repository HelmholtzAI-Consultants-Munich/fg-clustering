############################################
# imports
############################################

import numpy as np
from tqdm import tqdm

from sklearn_extra.cluster import KMedoids

import fgclustering.utils as utils
import fgclustering.plotting as plotting

############################################
# Forest-guided Clustering
############################################


def forest_guided_clustering(output, data, target_column, model,   
                             max_K = 6, number_of_clusters = None, max_iter_clustering = 500,  
                             bootstraps_JI = 300, discart_value_JI = 0.6,
                             bootstraps_p_value = 10000, thr_pvalue = 0.05, random_state = 42):
    '''Run forest-guided clustering algirthm for Random Forest Classifier or Regressor. The optimal number of clusters 
    for a k-medoids clustering is computed, based on the distance matrix coputed from the Random Forest proximity matrix. 
    Features are ranked and filtered based on staistical tests (ANOVA for continuous featres, chi square for categorical features).
    Feature distribution per cluster is shown in a heatmap and boxplots. Feature importance is plotted to show 
    the importance of each feature for each cluster, measured by variance and impurity of the feature within the cluster, 
    i.e. the higher the feature importance, the lower the feature variance / impurity within the cluster.

    :param output: [description]
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
    :return: Number of optimal clusters.
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
        k = optimizeK(distanceMatrix, y.to_numpy(), max_K, bootstraps_JI, max_iter_clustering, discart_value_JI, method, random_state)
    else:
        k = number_of_clusters
        
    print(f"Visualizing forest guided clustering for {k} clusters")
    
    plotting.plot_forest_guided_clustering(output, X, y, method, distanceMatrix, k, thr_pvalue, bootstraps_p_value, random_state)
    
    return k


def optimizeK(distance_matrix, y, max_K, bootstraps, max_iter_clustering, discart_value, method, random_state):
    """
    Compute the optimal number of clusters for k-medoids clustering (trade-off between cluster purity and cluster stability) . 
    Parameters
    ----------
        distanceMatrix: Pandas DataFrame
            Distance matrix computed from Random Forest proximity matrix.
        y: Pandas Series
            Target column.
        max_K: int
            Maximum number of clusters for cluster score computation.
        bootstraps: int
            Number of bootstraps to compute the Jaccard Index.
        max_iter_clustering: int
            Number of iterations for k-medoids clustering.
        discart_value: float
            Minimum Jaccard Index for cluster stability.
        method: string
            Model type of Random Forest model: classifier or regression.
        random_state: int
            Seed number for random state.

    Returns
    -------
        optimal_k: int
            Optimal number of clusters.
    """
    
    score_min = np.inf
    optimal_k = 1
    
    for k in tqdm(range(2, max_K)):
        #compute clusters        
        cluster_method = lambda X: KMedoids(n_clusters=k, random_state=random_state, init = 'build', method = "pam", max_iter=max_iter_clustering).fit(X).labels_
        labels = cluster_method(distance_matrix)

        # compute jaccard indices
        index_per_cluster = _compute_stability_indices(distance_matrix, cluster_method, bootstraps, random_state)
        min_index = min([index_per_cluster[cluster] for cluster in index_per_cluster.keys()])
        
        # only continue if jaccard indices are all larger 0.6 (thus all clusters are stable)
        print('For number of cluster {} the Jaccard Index is {}'.format(k, min_index))
        if min_index > discart_value:
            if method == "classifier":
                # compute balanced purities
                score = compute_balanced_average_impurity(y, labels)
            elif method == "regression":
                # compute the total within cluster variation
                score = compute_total_within_cluster_variation(y, labels)
            if score<score_min:
                optimal_k = k
                score_min = score
            print('For number of cluster {} the score is {}'.format(k,score))
        else:
            print('Clustering is instable, no score computed!')
            
    print(f"The optimal number of clusters is {optimal_k}")

    return optimal_k



def _compute_stability_indices(distance_matrix, cluster_method, bootstraps, random_state):
    """
    Compute stability of each cluster via Jaccard Index of bootstraped vs original clustering. 
    Parameters
    ----------
        distanceMatrix: Pandas DataFrame
            Distance matrix computed from Random Forest proximity matrix.
        cluster_method: sklearn_extra object
            K-medoids instance.
        bootstraps: int
            Number of bootstraps to compute the Jaccard Index.
        random_state: int
            Seed number for random state.

    Returns
    -------
        index_per_cluster: dict
            Dictionary with cluster labels as keys and Jaccard Indices as values.
    """
    np.random.seed = random_state
    
    matrix_shape = distance_matrix.shape
    assert len(matrix_shape) == 2, "error distance_matrix is not a matrix"
    assert matrix_shape[0] == matrix_shape[1], "error distance matrix is not square"
    
    labels = cluster_method(distance_matrix)
    clusters = np.unique(labels)
    number_datapoints = len(labels)
    index_vector = np.arange(number_datapoints)
    
    indices_original_clusters = _translate_cluster_labels_to_dictionary_of_index_sets_per_cluster(labels)
    index_per_cluster = {cluster: 0 for cluster in clusters}
    
    for i in range(bootstraps):
        bootstrapped_distance_matrix, mapping_bootstrapped_indices_to_original_indices = _bootstrap_matrix(distance_matrix)
        bootstrapped_labels = cluster_method(bootstrapped_distance_matrix)
        
        # now compute the indices for the different clusters
        indices_bootstrap_clusters = _translate_cluster_labels_to_dictionary_of_index_sets_per_cluster(bootstrapped_labels, mapping = mapping_bootstrapped_indices_to_original_indices)
        jaccard_matrix = _compute_jaccard_matrix(clusters, indices_bootstrap_clusters, indices_original_clusters)
        
        # compute optimal jaccard index for each cluster -> choose maximum possible jaccard index first
        for cluster_round in range(len(jaccard_matrix)):
            best_index = jaccard_matrix.max(axis=1).max()       
            original_cluster_number = jaccard_matrix.max(axis=1).argmax()
            bootstrapped_cluster_number = jaccard_matrix[original_cluster_number].argmax()
            jaccard_matrix[original_cluster_number] = -np.inf
            jaccard_matrix[:,bootstrapped_cluster_number] = -np.inf

            original_cluster = clusters[original_cluster_number]
            index_per_cluster[original_cluster] += best_index
                                    
    # normalize
    index_per_cluster = {cluster: index_per_cluster[cluster]/bootstraps for cluster in clusters}
        
    return index_per_cluster



def _translate_cluster_labels_to_dictionary_of_index_sets_per_cluster(labels, mapping = False):
    """
    Create dictionary that maps indices to cluster labels. 
    Parameters
    ----------
        labels: numpy array
            Clustering labels.
        mapping: Bool
            Mapping of bootstrapped to original indices.

    Returns
    -------
        indices_clusters: dict
            Dictionary with cluster labels as keys and index of instances that belong to the respective cluster as labels.
    """
    
    clusters = np.unique(labels)
    number_datapoints = len(labels)
    index_vector = np.arange(number_datapoints)
    
    indices_clusters = {}
    for cluster in clusters:
        indices = set(index_vector[labels == cluster])
        if mapping is not False:
            #translate from the bootstrapped indices to the original naming of the indices
            indices = set([mapping[index] for index in indices])

        indices_clusters[cluster] = indices

    return indices_clusters



def _bootstrap_matrix(M):
    """
    Create a bootstrap from the original matrix. 
    Parameters
    ----------
        M: Pandas DataFrame
            Original matrix.

    Returns
    -------
        M_bootstrapped: Pandas DataFrame
            Bootstrapped matrix.
        mapping_bootstrapped_indices_to_original_indices: dict
            Mapping from bootstrapped to original indices.
    """
    
    lm = len(M)
    bootstrapped_samples = np.random.choice(np.arange(lm), lm)
    M_bootstrapped = M[:,bootstrapped_samples][bootstrapped_samples,:]
    
    mapping_bootstrapped_indices_to_original_indices = {bootstrapped : original for bootstrapped, original in enumerate(bootstrapped_samples)}
    
    return M_bootstrapped, mapping_bootstrapped_indices_to_original_indices



def _compute_jaccard_matrix(clusters, indices_bootstrap_clusters, indices_original_clusters):
    """
    Compute Jaccard Index between all possible cluster combinations of original vs bootstrapped clustering. 
    Parameters
    ----------
        clusters: numpy array
            Clustering labels.
        indices_bootstrap_clusters: dict
            Dictionary with cluster labels as keys and index of instances that belong to the respective cluster as labels for boostrapped clustering.
        indices_original_clusters: dict
            Dictionary with cluster labels as keys and index of instances that belong to the respective cluster as labels for original clustering.

    Returns
    -------
        jaccard_matrix: numpy array
            Jaccard Index for all cluster combinations.
    """
    
    indices_bootstrap_all = np.unique([index for i, cluster_bootstrap in enumerate(clusters) for index in indices_bootstrap_clusters[cluster_bootstrap]])
    jaccard_matrix = np.zeros([len(clusters), len(clusters)])

    for i, cluster_original in enumerate(clusters):
        for j, cluster_bootstrap in enumerate(clusters):
            indices_bootstrap = indices_bootstrap_clusters[cluster_bootstrap]
            indices_original = indices_original_clusters[cluster_original]
            
            # only compute overlap for instances that were in the whole bootstrap sample
            indices_original = indices_original.intersection(indices_bootstrap_all)

            intersection = indices_original.intersection(indices_bootstrap)
            union = indices_original.union(indices_bootstrap)
            
            jaccard_matrix[i,j] = len(intersection)/len(union)
            
    return jaccard_matrix
