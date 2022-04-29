############################################
# imports
############################################

import numpy as np
from tqdm import tqdm

from sklearn_extra.cluster import KMedoids
from joblib import Parallel, delayed

import fgclustering.statistics as statistics

############################################
# Optimize number of clusters k
############################################


def _compute_jaccard_matrix(clusters, indices_bootstrap_clusters, indices_original_clusters):
    '''Compute Jaccard Index between all possible cluster combinations of original vs bootstrapped clustering.

    :param clusters: Clustering labels.
    :type clusters: numpy.ndarray
    :param indices_bootstrap_clusters: Dictionary with cluster labels as keys and index of instances that 
        belong to the respective cluster as labels for boostrapped clustering.
    :type indices_bootstrap_clusters: dict
    :param indices_original_clusters: Dictionary with cluster labels as keys and index of instances that 
        belong to the respective cluster as labels for original clustering.
    :type indices_original_clusters: dict
    :return: Jaccard Index for all cluster combinations.
    :rtype: numpy.ndarray
    '''
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


def _bootstrap_matrix(M):
    '''Create a bootstrap from the original matrix.

    :param M: Original matrix.
    :type M: pandas.DataFrame
    :return: M_bootstrapped: ootstrapped matrix; 
        mapping_bootstrapped_indices_to_original_indices: mapping from bootstrapped to original indices.
    :rtype: pandas.DataFrame, dict
    '''    
    lm = len(M)
    bootstrapped_samples = np.random.choice(np.arange(lm), lm)
    M_bootstrapped = M[:,bootstrapped_samples][bootstrapped_samples,:]
    
    mapping_bootstrapped_indices_to_original_indices = {bootstrapped : original for bootstrapped, original in enumerate(bootstrapped_samples)}
    
    return M_bootstrapped, mapping_bootstrapped_indices_to_original_indices


def _translate_cluster_labels_to_dictionary_of_index_sets_per_cluster(labels, mapping = False):
    '''Create dictionary that maps indices to cluster labels. 

    :param labels: Clustering labels.
    :type labels: numpy.ndarray
    :param mapping: Mapping of bootstrapped to original indices, defaults to False
    :type mapping: bool, optional
    :return: Dictionary with cluster labels as keys and index of instances that belong to the respective cluster as labels.
    :rtype: dict
    '''
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


def _compute_stability_indices(distance_matrix, cluster_method, labels, bootstraps, random_state):
    '''Compute stability of each cluster via Jaccard Index of bootstraped vs original clustering.

    :param distance_matrix: Proximity matrix of Random Forest model.
    :type distance_matrix: pandas.DataFrame
    :param cluster_method: Lambda function wrapping the k-mediods clustering function.
    :type cluster_method: object
    :param bootstraps: Number of bootstraps to compute the Jaccard Index, defaults to 300
    :type bootstraps: int
    :param random_state: Seed number for random state, defaults to 42
    :type random_state: int
    :return: Dictionary with cluster labels as keys and Jaccard Indices as values.
    :rtype: dict
    '''
    np.random.seed = random_state
    
    matrix_shape = distance_matrix.shape
    assert len(matrix_shape) == 2, "error distance_matrix is not a matrix"
    assert matrix_shape[0] == matrix_shape[1], "error distance matrix is not square"

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


def _optimizeKloop(k, distance_matrix, y, random_state, max_iter_clustering, bootstraps, discart_value, method):
    '''
    TBA
    :param k:
    :param distance_matrix:
    :param random_state:
    :param max_iter_clustering:
    :param bootstraps:
    :param discart_value:
    :return:
    '''
    # compute clusters
    print(f'Checking number of clusters k={k}')
    cluster_method = lambda X: KMedoids(n_clusters=k, random_state=random_state, init='build', method="pam",
                                        max_iter=max_iter_clustering).fit(X).labels_
    labels = cluster_method(distance_matrix)

    # compute jaccard indices
    index_per_cluster = _compute_stability_indices(distance_matrix, cluster_method, labels, bootstraps, random_state)
    min_index = min([index_per_cluster[cluster] for cluster in index_per_cluster.keys()])

    # only continue if jaccard indices are all larger 0.6 (thus all clusters are stable)
    print('For number of cluster {} the Jaccard Index is {}'.format(k, min_index))
    if min_index > discart_value:
        if method == "classifier":
            # compute balanced purities
            score = statistics.compute_balanced_average_impurity(y, labels)
        elif method == "regression":
            # compute the total within cluster variation
            score = statistics.compute_total_within_cluster_variation(y, labels)
        print('For number of cluster {} the score is {}'.format(k, score))
        return {k: score}
    else:
        print('Clustering is instable, no score computed!')
        return {k: np.nan}


def optimizeK(distance_matrix, y, max_K, bootstraps, max_iter_clustering, discart_value, method, random_state):
    '''Compute the optimal number of clusters for k-medoids clustering (trade-off between cluster purity and cluster stability). 

    :param distance_matrix: Proximity matrix of Random Forest model.
    :type distance_matrix: pandas.DataFrame
    :param y: Target column.
    :type y: pandas.Series
    :param max_K: Maximum number of clusters for cluster score computation, defaults to 6
    :type max_K: int
    :param bootstraps: Number of bootstraps to compute the Jaccard Index, defaults to 300
    :type bootstraps: int
    :param max_iter_clustering: Number of iterations for k-medoids clustering, defaults to 500
    :type max_iter_clustering: int
    :param discart_value: Minimum Jaccard Index for cluster stability, defaults to 0.6
    :type discart_value: float
    :param method: Model type of Random Forest model: classifier or regression.
    :type method: str
    :param random_state: Seed number for random state, defaults to 42
    :type random_state: int
    :return: Optimal number of clusters.
    :rtype: int
    '''

    results = Parallel(n_jobs=2)(
        delayed(_optimizeKloop)(k, distance_matrix, y, random_state, max_iter_clustering, bootstraps, discart_value, method) for k
        in range(2, max_K))
    # Flat to dictionary:
    results = dict(sum(map(list, map(dict.items, results)), []))
    optimal_k, score = min(results.items(), key=lambda k: k[1])  # optimal k is the one with minimum impurity/within cluster variation score

    return optimal_k

