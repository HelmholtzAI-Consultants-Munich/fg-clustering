############################################
# imports
############################################

import numpy as np
from tqdm import tqdm

from sklearn_extra.cluster import KMedoids
from joblib import Parallel, delayed
import collections, functools, operator
from numba import njit, prange

import fgclustering.statistics as statistics

import warnings
warnings.filterwarnings('ignore')

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


@njit
def _filter_bootstrap(M, bootstrapped_samples):
    n = M.shape[0]

    M_bootstrapped = np.empty((n,n), dtype=M.dtype).T # transpose to get F-contiguous

    for j in prange(n):
        M_bootstrapped[:, j] = M[:,bootstrapped_samples[j]][bootstrapped_samples]
    
    return M_bootstrapped


def _bootstrap_matrix(M):
    '''Create a bootstrap from the original matrix.

    :param M: Original matrix.
    :type M: pandas.DataFrame
    :return: M_bootstrapped: ootstrapped matrix; 
        mapping_bootstrapped_indices_to_original_indices: mapping from bootstrapped to original indices.
    :rtype: pandas.DataFrame, dict
    '''
    #np.random.seed(seed)
    lm = len(M)
    bootstrapped_samples = np.random.choice(np.arange(lm), lm)
    bootstrapped_samples = np.sort(bootstrapped_samples) #Sort samples to increase speed. Does not affect downstream analysis because M is symmetric
    M_bootstrapped = _filter_bootstrap(M, bootstrapped_samples)
    mapping_bootstrapped_indices_to_original_indices = {bootstrapped : original for bootstrapped, original in enumerate(bootstrapped_samples)}
   
    return M_bootstrapped, mapping_bootstrapped_indices_to_original_indices


def _bootstrap_matrix_old(M, seed=42):
    '''Create a bootstrap from the original matrix.

    :param M: Original matrix.
    :type M: pandas.DataFrame
    :return: M_bootstrapped: ootstrapped matrix; 
        mapping_bootstrapped_indices_to_original_indices: mapping from bootstrapped to original indices.
    :rtype: pandas.DataFrame, dict
    '''
    np.random.seed(seed)
    lm = len(M)
    bootstrapped_samples = np.random.choice(np.arange(lm), lm)
    bootstrapped_samples = np.sort(bootstrapped_samples) #Sort samples to increase speed. Does not affect downstream analysis because M is symmetric
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


def _compute_stability_indices(distance_matrix, cluster_method, clusters, indices_original_clusters):
    '''Function that parallelizes the bootstrapping loop in the _compute_stability_indices function. 
    Compute stability of each cluster via Jaccard Index of original clustering vs clustering of one bootstraped sample. 

    :param distance_matrix: Proximity matrix of Random Forest model.
    :type distance_matrix: pandas.DataFrame
    :param cluster_method: Lambda function wrapping the k-mediods clustering function.
    :type cluster_method: object
    :param clusters: possible clusters (unique cluster labels)
    :type clusters: numpy array
    :param indices_original_clusters:  dictionary that maps indices to cluster labels
    :type indices_original_clusters: dict
    :return: Dictionary with Jaccard scores for each cluster in the given boostrap sample
    :rtype: dict
    '''
    index_per_cluster = {cluster: 0 for cluster in clusters}
    
    bootstrapped_distance_matrix, mapping_bootstrapped_indices_to_original_indices = _bootstrap_matrix(distance_matrix)
    bootstrapped_labels = cluster_method(bootstrapped_distance_matrix)

    # now compute the indices for the different clusters
    indices_bootstrap_clusters = _translate_cluster_labels_to_dictionary_of_index_sets_per_cluster(bootstrapped_labels, mapping=mapping_bootstrapped_indices_to_original_indices)
    jaccard_matrix = _compute_jaccard_matrix(clusters, indices_bootstrap_clusters, indices_original_clusters)

    # compute optimal jaccard index for each cluster -> choose maximum possible jaccard index first
    for cluster_round in range(len(jaccard_matrix)):
        best_index = jaccard_matrix.max(axis=1).max()
        original_cluster_number = jaccard_matrix.max(axis=1).argmax()
        bootstrapped_cluster_number = jaccard_matrix[original_cluster_number].argmax()
        jaccard_matrix[original_cluster_number] = -np.inf
        jaccard_matrix[:, bootstrapped_cluster_number] = -np.inf

        original_cluster = clusters[original_cluster_number]
        index_per_cluster[original_cluster] += best_index

    return index_per_cluster


def _compute_stability_indices_parallel(distance_matrix, labels, cluster_method, bootstraps, n_jobs):
    '''Compute stability of each cluster via Jaccard Index of bootstraped vs original clustering.

    :param distance_matrix: Proximity matrix of Random Forest model.
    :type distance_matrix: pandas.DataFrame
    :param labels: original cluster labels
    :type labels: numpy array
    :param cluster_method: Lambda function wrapping the k-mediods clustering function.
    :type cluster_method: object
    :param bootstraps: Number of bootstraps to compute the Jaccard Index, defaults to 300
    :type bootstraps: int
    :param n_jobs: number of jobs to run in parallel when computing the cluster stability. n_jobs=1 means no parallel computing is used, defaults to 1
    :type n_jobs: int, optional
    :return: Dictionary with cluster labels as keys and Jaccard Indices as values.
    :rtype: dict
    '''
    clusters = np.unique(labels)
    indices_original_clusters = _translate_cluster_labels_to_dictionary_of_index_sets_per_cluster(labels)

    # Compute Jaccard Index per bootstrapped sample
    index_per_cluster = Parallel(n_jobs=n_jobs)(
        delayed(_compute_stability_indices)(distance_matrix, cluster_method, clusters, indices_original_clusters) for i in range(bootstraps))
    # Sum Jaccard values of the same keys across dictionaries
    index_per_cluster = dict(functools.reduce(operator.add,
                                   map(collections.Counter, index_per_cluster)))
    # normalize:
    index_per_cluster = {cluster: index_per_cluster[cluster]/bootstraps for cluster in clusters}

    return index_per_cluster
    
    
def optimizeK(distance_matrix, y, model_type, max_K, method_clustering, init_clustering, max_iter_clustering, discart_value_JI, bootstraps_JI, random_state, n_jobs):
    '''Compute the optimal number of clusters for k-medoids clustering (trade-off between cluster purity and cluster stability). 

    :param distance_matrix: Proximity matrix of Random Forest model.
    :type distance_matrix: pandas.DataFrame
    :param y: Target column.
    :type y: pandas.Series
    :param model_type: Model type of Random Forest model: classifier or regression.
    :type model_type: str
    :param max_K: Maximum number of clusters for cluster score computation, defaults to 6
    :type max_K: int
    :param method_clustering: Which algorithm to use. 'alternate' is faster while 'pam' is more accurate, defaults to 'pam'
    :type method_clustering: {'alternate', 'pam'}, optional
    :param init_clustering: Specify medoid initialization method. To speed up computation for large datasets use 'random'.
        See sklearn documentation for parameter description, defaults to 'k-medoids++'
    :type init_clustering: {'random', 'heuristic', 'k-medoids++', 'build'}, optional
    :param max_iter_clustering: Number of iterations for k-medoids clustering, defaults to 500
    :type max_iter_clustering: int
    :param discart_value: Minimum Jaccard Index for cluster stability, defaults to 0.6
    :type discart_value: float
    :param bootstraps_JI: Number of bootstraps to compute the Jaccard Index, defaults to 300
    :type bootstraps_JI: int
    :param random_state: Seed number for random state, defaults to 42
    :type random_state: int
    :param n_jobs: number of jobs to run in parallel when computing the cluster stability. 
        n_jobs=1 means no parallel computing is used, defaults to 1
    :type n_jobs: int, optional
    :return: Optimal number of clusters.
    :rtype: int
    '''  
    np.random.seed(random_state)
    
    # Check distance matrix
    matrix_shape = distance_matrix.shape
    assert len(matrix_shape) == 2, "error distance_matrix is not a matrix"
    assert matrix_shape[ 0 ] == matrix_shape[ 1 ], "error distance matrix is not square"
    
    score_min = np.inf
    optimal_k = 1
    
    for k in tqdm(range(2, max_K)):
        #compute clusters        
        cluster_method = lambda X: KMedoids(n_clusters=k, random_state=random_state, init=init_clustering, method=method_clustering, max_iter=max_iter_clustering).fit(X).labels_
        labels = cluster_method(distance_matrix)

        # compute jaccard indices
        index_per_cluster = _compute_stability_indices_parallel(distance_matrix, labels, cluster_method, bootstraps_JI, n_jobs)
        min_index = min([index_per_cluster[cluster] for cluster in index_per_cluster.keys()])
        
        # only continue if jaccard indices are all larger 0.6 (thus all clusters are stable)
        print('For number of cluster {} the Jaccard Index is {}'.format(k, min_index))
        if min_index > discart_value_JI:
            if model_type == "classifier":
                # compute balanced purities
                score = statistics.compute_balanced_average_impurity(y, labels)
            elif model_type == "regression":
                # compute the total within cluster variation
                score = statistics.compute_total_within_cluster_variation(y, labels)
            if score<score_min:
                optimal_k = k
                score_min = score
            print('For number of cluster {} the score is {}'.format(k,score))
        else:
            print('Clustering is instable, no score computed!')

    return optimal_k
