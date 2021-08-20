import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm

def forest_guided_clustering(rf, data, target_column, max_K = 6, 
                             random_state=42, 
                             bootstraps = 300, 
                             max_iter_clustering = 300):
    y = data.loc[:,target_column].to_numpy()
    X = data.drop(columns=[target_column]).to_numpy()
    
    distanceMatrix = 1 - proximityMatrix(model, X)
    k = optimizeK(distanceMatrix, X, y, max_K = 6,
                  random_state=random_state, 
                  bootstraps = bootstraps,
                  max_iter_clustering = max_iter_clustering)
    
    plot_forest_guided_clustering(rf, data, target_column, k, random_state = random_state)

def optimizeK(distance_matrix, x, y,
              max_K = 6, 
              random_state=42, 
              discart_value = 0.6,
              bootstraps = 300, 
              max_iter_clustering = 300):
    
    min_purity = 1
    optimal_k = 1
    
    for k in tqdm(range(2, max_K)):
        #compute clusters        
        cluster_method = lambda X: KMedoids(n_clusters=k, 
                                            random_state=random_state, init = 'build',
                                            method = "pam", max_iter=max_iter_clustering).fit(X).labels_
        labels = cluster_method(distance_matrix)

        # compute jaccard indices
        index_per_cluster = compute_stability_indices(distance_matrix, 
                                                      cluster_method = cluster_method, 
                                                      seed = random_state, 
                                                      bootstraps = bootstraps)
        min_index = min([index_per_cluster[cluster] for cluster in index_per_cluster.keys()])
        print(f'k: {k} index: {min_index}')
        
        # only continue if jaccard indices are all larger 0.6 (thus all clusters are stable)
        if min_index > discart_value:
            # compute balanced purities
            balanced_purity = compute_balanced_average_purity(y, labels)

            if balanced_purity<min_purity:
                optimal_k = k
                min_purity = balanced_purity
    return optimal_k
        
def compute_balanced_average_purity(y, labels):
    n0 = sum(y==0)
    n1 = sum(y==1)
    
    if n0<=n1:
        small_label = 0
        large_label = 1
        up_scaling_factor = n1/n0
    else:
        small_label = 1
        large_label = 0
        up_scaling_factor = n0/n1
    
    balanced_purities = []
    for cluster in np.unique(labels):
        y_cluster = y[labels == cluster]
        
        x_small = sum(y_cluster == small_label)*up_scaling_factor
        x_large = sum(y_cluster == large_label)
        x_tot = x_small+x_large
        balanced_purity = (x_small/x_tot)*(x_large/x_tot)
        normalized_balanced_purity = balanced_purity/0.25
        
        balanced_purities.append(normalized_balanced_purity)
    
    average_balanced_purities = np.mean(balanced_purities)
    return average_balanced_purities


def compute_stability_indices(distance_matrix, cluster_method, bootstraps = 300, seed = 42):
    matrix_shape = distance_matrix.shape
    assert len(matrix_shape) == 2, "error distance_matrix is not a matrix"
    assert matrix_shape[0] == matrix_shape[1], "error distance matrix is not square"
    np.random.seed = seed
    
    labels = cluster_method(distance_matrix)
    clusters = np.unique(labels)
    number_datapoints = len(labels)
    index_vector = np.arange(number_datapoints)
    
    indices_original_clusters = _translate_cluster_labels_to_dictionary_of_index_sets_per_cluster(labels)
    
    index_per_cluster = {cluster: 0 for cluster in clusters}
    
    for i in range(bootstraps):
        bootsrapped_distance_matrix, mapping_bootstrapped_indices_to_original_indices = bootstrap_matrix(distance_matrix)
        bootstrapped_labels = cluster_method(bootsrapped_distance_matrix)
        # now compute the indices for the different clusters
        indices_bootstrap_clusters = _translate_cluster_labels_to_dictionary_of_index_sets_per_cluster(bootstrapped_labels, 
                                                                                                       mapping = mapping_bootstrapped_indices_to_original_indices)
        
        
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

def bootstrap_matrix(M):
    lm = len(M)
    bootstrapped_samples = np.random.choice(np.arange(lm), lm)
    M_bootstrapped = M[:,bootstrapped_samples][bootstrapped_samples,:]
    
    mapping_bootstrapped_indices_to_original_indices = {bootstrapped : original for bootstrapped, original in enumerate(bootstrapped_samples)}
    
    return M_bootstrapped, mapping_bootstrapped_indices_to_original_indices

def proximityMatrix(model, X, normalize=True):  
    '''computes the proximity matrix from the model'''

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


def _translate_cluster_labels_to_dictionary_of_index_sets_per_cluster(labels, mapping = False):

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

def _compute_jaccard_matrix(clusters, indices_bootstrap_clusters, indices_original_clusters):
        
    jaccard_matrix = np.zeros([len(clusters), len(clusters)])
    for i, cluster_original in enumerate(clusters):
        for j, cluster_bootstrap in enumerate(clusters):
            indices_bootstrap = indices_bootstrap_clusters[cluster_bootstrap]
            indices_original = indices_original_clusters[cluster_original]

            intersection = indices_original.intersection(indices_bootstrap)
            union = indices_original.union(indices_bootstrap)

            jaccard_matrix[i,j] = len(intersection)/len(union)
            
    return jaccard_matrix