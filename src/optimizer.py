############################################
# imports
############################################

import numpy as np
from tqdm import tqdm

from sklearn_extra.cluster import KMedoids

############################################
# Optimize number of clusters k
############################################


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
        index_per_cluster = compute_stability_indices(distance_matrix, cluster_method, bootstraps, random_state)
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



def compute_stability_indices(distance_matrix, cluster_method, bootstraps, random_state):
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
            indices = [mapping[index] for index in indices]

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



def compute_balanced_average_impurity(y, labels):
    """
    Compute balanced average impurity as score for Random Forest classifier clustering. 
    Impurity score is an Gini Coefficient of the classes within each cluster. The class sizes are thereby balanced by
    by rescaling with the inverse size of the class in the overall dataset.
    
    Parameters
    ----------
        y: Pandas Series
            Target column.
        labels: numpy array
            Clustering labels.

    Returns
    -------
        score: float
            Score for clustering defined by balanced average purity .
    """
    
    #compute the number of datapoints for each class to use it then for rescaling of the 
    #class sizes within each cluster
    class_rescaling_factor = {class_: 1/sum(y==class_) for class_ in np.unique(y)} #rescaling with inverse class size
        
    balanced_impurities = []
    
    for cluster in np.unique(labels):
        y_cluster = y[labels == cluster]
        
        #compute balanced class probabilities (rescaled with overall class size)
        class_probabilities_unnormalized = [sum(y_cluster==class_)*class_rescaling_factor[class_] for class_ in np.unique(y)]
        class_probabilities_unnormalized = np.array(class_probabilities_unnormalized)
        normalization_factor = class_probabilities_unnormalized.sum()
        class_probabilities = class_probabilities_unnormalized / normalization_factor
        
        #compute (balanced) gini impurity
        gini_impurity = 1 - np.sum(class_probabilities**2)
        
        balanced_impurities.append(gini_impurity)
    
    score = np.mean(balanced_impurities)
    
    return score



def compute_total_within_cluster_variation(y, labels):
    """
    Compute total within cluster variation as score for Random Forest regression clustering. 
    Parameters
    ----------
        y: Pandas Series
            Target column.
        labels: numpy array
            Clustering labels.

    Returns
    -------
        score: float
            Score for clustering defined by total within cluster variation.
    """
    
    score = 0
    for cluster in np.unique(labels):
        y_cluster = y[labels == cluster]
        score += np.var(y_cluster)*len(y_cluster)
        
    return score
    
    
    
