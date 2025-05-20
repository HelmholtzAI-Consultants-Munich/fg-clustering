############################################
# imports
############################################

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from pandas.api.types import is_numeric_dtype, is_categorical_dtype


############################################
# functions
############################################

def compute_balanced_average_impurity(categorical_values, cluster_labels, rescaling_factor=None):
    """Compute balanced average impurity as score for categorical values in a clustering.
    Impurity score is a Gini Coefficient of the classes within each cluster.
    The class sizes are balanced by rescaling with the inverse size of the class in the overall dataset.

    :param categorical_values: Values of categorical feature / target.
    :type categorical_values: pandas.Series
    :param cluster_labels: Cluster labels for each value.
    :type cluster_labels: numpy.ndarray
    :param rescaling_factor: Dictionary with rescaling factor for each class / unique feature value.
        If parameter is set to None, the rescaling factor will be computed from the input data categorical_values, defaults to None
    :type rescaling_factor: dict
    :return: Impurity score.
    :rtype: float
    """
    # compute the number of datapoints for each class to use it then for rescaling of the
    # class sizes within each cluster --> rescaling with inverse class size
    if rescaling_factor is None:
        rescaling_factor = {
            class_: 1 / sum(categorical_values == class_) for class_ in np.unique(categorical_values)
        }
    balanced_impurities = []

    for cluster in np.unique(cluster_labels):
        categorical_values_cluster = categorical_values[cluster_labels == cluster]

        # compute balanced class probabilities (rescaled with overall class size)
        class_probabilities_unnormalized = [
            sum(categorical_values_cluster == class_) * rescaling_factor[class_]
            for class_ in np.unique(categorical_values)
        ]
        class_probabilities_unnormalized = np.array(class_probabilities_unnormalized)
        normalization_factor = class_probabilities_unnormalized.sum()
        class_probabilities = class_probabilities_unnormalized / normalization_factor

        # compute (balanced) gini impurity
        gini_impurity = 1 - np.sum(class_probabilities**2)
        balanced_impurities.append(gini_impurity)

    score = np.mean(balanced_impurities)

    return score


def compute_total_within_cluster_variation(continuous_values, cluster_labels):
    """Compute total within cluster variation as score for continuous values in a clustering.

    :param continuous_values: Values of continuous feature / target.
    :type continuous_values: pandas.Series
    :param cluster_labels: Cluster labels for each value.
    :type cluster_labels: numpy.ndarray
    :return: Within cluster variation score.
    :rtype: float
    """
    score = 0
    for cluster in np.unique(cluster_labels):
        continuous_values_cluster = continuous_values[cluster_labels == cluster]
        score += np.var(continuous_values_cluster) * len(continuous_values_cluster)

    return score

def calculate_feature_cluster_distances(X, clusters, distance_func, scale=False, verbose=False):
    """
    Calculate distance between overall feature distribution and cluster-wise feature distributions.
    Supports numeric and categorical features, using a specified distance function (e.g. Wasserstein, Jensen-Shannon).
    Distances are normalized per cluster to [0, 1] range.
    
    :param X: Feature matrix (pandas DataFrame or convertible object).
    :type X: pd.DataFrame or array-like
    :param clusters: Array of cluster labels, same length as number of rows in X.
    :type clusters: array-like
    :param distance_func: Callable distance function that takes (all_values, cluster_values, is_categorical) and returns (distance, meta).
    :type distance_func: function
    :param scale: Whether to scale numeric features by their standard deviation before distance computation, defaults to False.
    :type scale: bool
    :param verbose: Whether to print detailed output during processing, defaults to False.
    :type verbose: bool
    :return: DataFrame of normalized distances with features as rows and clusters as columns.
    :rtype: pd.DataFrame
    """
    
    # Ensure input is a dataframe
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    clusters_unique = np.unique(clusters)
    distances = pd.DataFrame(index=X.columns, columns=clusters_unique)
    X_proc = X.copy()
    
    # Optional scaling of numeric values
    if scale:
        for col in X_proc.columns:
            if is_numeric_dtype(X_proc[col]):
                std = X_proc[col].std()
                if std > 0:
                    X_proc[col] = X_proc[col] / std
    
    # Loop through columns
    for feature in X_proc.columns:
        # Extract all values of column
        values_background = X_proc[feature]
        
        if verbose:
            print(f"\nProcessing feature: {feature}")
        
        # Determine if column is numeric
        if is_numeric_dtype(values_background):
            if values_background.nunique() <= 1:
                if verbose:
                    print(f" - Skipping numeric feature with zero variance.")
                distances.loc[feature, :] = np.nan
                continue
            
            # Loop through cluster, extract cluster values
            for cluster in clusters_unique:
                values_cluster = X_proc.loc[clusters == cluster, feature]
                
                if values_cluster.empty:
                    if verbose:
                        print(f" - Cluster {cluster} is empty. Skipping.")
                    distances.loc[feature, cluster] = np.nan
                    continue
                
                # Calculate distance to background values
                dist, meta = distance_func(values_background, values_cluster, is_categorical=False)
                distances.loc[feature, cluster] = dist
                
                # Optionally print number of used bins
                if verbose and meta and "bins" in meta:
                    print(f" - Used {meta['bins']} bins for numeric feature.")
        
        # Handle categorical columns
        elif is_categorical_dtype(values_background) or values_background.dtype == "object":
            if values_background.nunique() <= 1:
                if verbose:
                    print(f" - Skipping categorical feature with only one level.")
                distances.loc[feature, :] = np.nan
                continue
            
            # Extract cluster values and calculate distance to background values
            for cluster in clusters_unique:
                values_cluster = X_proc.loc[clusters == cluster, feature]
                dist, _ = distance_func(values_background, values_cluster, is_categorical=True)
                distances.loc[feature, cluster] = dist
        
        # Handle unsupported data type
        else:
            if verbose:
                print(f" - Unsupported data type. Skipping feature.")
            distances.loc[feature, :] = np.nan

    # Divide the distance by max for each cluster to get max equal to one
    col_max = distances.max(axis=0)
    col_max[col_max == 0] = np.nan  # Avoid division by zero
    distances = distances.div(col_max, axis=1)
    
    return distances



def wasserstein_distance_func(values_all, values_cluster, is_categorical=False):
    """
    Compute the Wasserstein distance between two distributions.

    - For numeric data, computes the standard 1D Wasserstein distance.
    - For categorical data, one-hot encodes the values and computes Wasserstein distances per dummy column.

    :param values_all: Full (background) distribution.
    :param values_cluster: Cluster-specific distribution.
    :param is_categorical: Whether the data is categorical.
    :return: Maximum Wasserstein distance (for categorical) or scalar distance (for numeric), and optional metadata (None).
    """
    
    if is_categorical:
        dummies_all = pd.get_dummies(values_all, drop_first=True)
        dummies_cluster = pd.get_dummies(values_cluster, drop_first=True)
        dummies_all, dummies_cluster = dummies_all.align(dummies_cluster, join="outer", fill_value=0)

        distances = [
            wasserstein_distance(dummies_all[col], dummies_cluster[col])
            for col in dummies_all.columns
        ]
        return np.nanmax(distances), None
    else:
        return wasserstein_distance(values_all, values_cluster), None


def jensen_shannon_distance_func(values_all, values_cluster, is_categorical=False):
    """
    Compute the Jensen-Shannon distance between two distributions.

    - For categorical data, compares normalized frequency distributions.
    - For numeric data, applies the Freedman–Diaconis rule to bin data and compares histograms.

    :param values_all: Full (background) distribution.
    :param values_cluster: Cluster-specific distribution.
    :param is_categorical: Whether the data is categorical.
    :return: Jensen-Shannon distance (float), and metadata including number of bins used (for numeric).
    """
    
    if is_categorical:
        cats = values_all.unique()
        p_ref = values_all.value_counts(normalize=True).reindex(cats, fill_value=0)
        p_cluster = values_cluster.value_counts(normalize=True).reindex(cats, fill_value=0)
        return jensenshannon(p_ref, p_cluster), None
    else:
        # Compute number of bins using Freedman-Diaconis rule, enforcing sensible bounds
        range_val = values_all.max() - values_all.min()
        iqr = values_all.quantile(0.75) - values_all.quantile(0.25)
        n_obs = len(values_all)
        
        if range_val <= 0 or iqr <= 0 or n_obs <= 1:
            bins = 10
        else:
            bin_width = 2 * iqr / (n_obs ** (1 / 3))
            bin_estimate = int(np.ceil(range_val / bin_width))
            bins = max(1, min(bin_estimate, n_obs, 100))


        
        
        edges = np.percentile(values_all, np.linspace(0, 100, bins + 1))
        hist_ref, _ = np.histogram(values_all, bins=edges)
        hist_cluster, _ = np.histogram(values_cluster, bins=edges)

        p_ref = hist_ref / np.sum(hist_ref) if np.sum(hist_ref) > 0 else np.ones_like(hist_ref) / len(hist_ref)
        p_cluster = hist_cluster / np.sum(hist_cluster) if np.sum(hist_cluster) > 0 else np.ones_like(hist_cluster) / len(hist_cluster)

        return jensenshannon(p_ref, p_cluster), {"bins": bins}


def _sort_clusters_by_target(data_clustering_ranked, model_type):
    """Sort clusters by mean target values in clusters.

    :param data_clustering_ranked: Filtered and ranked data frame incl features, target and cluster numbers.
    :type data_clustering_ranked: pandas.DataFrame
    :param model_type: Model type of Random Forest model: classifier or regression.
    :type model_type: str
    :return: Filtered and ranked feature matrix with ordered clusters.
    :rtype: pandas.DataFrame
    """
    # When using a classifier, the target value is label encoded, such that we can sort the clusters by target values
    original_target = data_clustering_ranked["target"].copy()

    if model_type == "classification":
        data_clustering_ranked["target"] = data_clustering_ranked["target"].astype("category").cat.codes

    # Compute mean target values for each cluster and sort by mean values
    cluster_means = data_clustering_ranked.groupby(["cluster"])[["cluster", "target"]].mean()
    cluster_means = cluster_means.sort_values(by="target").index

    # Map the sorted clusters to a new order,  replace clusters with the new mapping and ensure the 'cluster' column is a categorical type with ordered levels
    mapping = {cluster: i + 1 for i, cluster in enumerate(cluster_means)}
    data_clustering_ranked["cluster"] = pd.Categorical(
        data_clustering_ranked["cluster"].map(mapping), ordered=True
    )

    # Restore the original target values
    data_clustering_ranked["target"] = original_target

    return data_clustering_ranked


def calculate_feature_importance(X, y, clusters, distance_func="wasserstein", model_type="", scale=False, verbose=False):
    """Calculate importance of each feature within each cluster and then over all clusters. 
    
    :param X: Feature matrix.
    :type X: pandas.DataFrame
    :param y: Target variable.
    :type y: pandas.Series
    :param clusters: Cluster labels.
    :type clusters: array-like
    :param distance_func: Defines which distance should be calculated. Possible values: 'wasserstein', 'jensen-shannon'. 
                        Wasserstein is primarily built for continuous features, Jensen-Shannon for categorical features. 
    :type distance_func: str
    :param model_type: Type of model used to determine sorting order of target variable in clusters (e.g., "regression" or "classification").
    :type model_type: str
    :param scale: Whether to scale numeric features by their standard deviation - only in case of Wasserstein. 
    :type scale: bool
    :param verbose: Whether to print details during processing.
    :type verbose: bool
    :return: 
        - feature_importance_local (pd.DataFrame): Feature distances per cluster.
        - feature_importance_global (pd.Series): Mean importance across all clusters.
        - data_clustering_sorted (pd.DataFrame): Clustered and sorted dataset.
    :rtype: Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
    """

    data_clustering = pd.concat([X, y.rename("target"), pd.Series(clusters, name="cluster")], axis=1)

    # Calculate distances
    if distance_func == "wasserstein":
        feature_importance_local = calculate_feature_cluster_distances(
            X=X, 
            clusters=clusters, 
            distance_func=wasserstein_distance_func, 
            scale=scale, 
            verbose=verbose
        )
    elif distance_func == "jensen-shannon":
        feature_importance_local = calculate_feature_cluster_distances(
            X=X, 
            clusters=clusters, 
            distance_func=jensen_shannon_distance_func, 
            scale=False, # No scaling in case of Jensen-Shannon
            verbose=verbose
        )
    else:
        raise ValueError("Invalid distance_func. Choose 'wasserstein' or 'jensen-shannon'.")

    # Aggregate over all clusters
    feature_importance_global = feature_importance_local.mean(axis=1)
    # Sort features by mean and extract names
    sorted_feature_names = feature_importance_global.sort_values(ascending=False).index.tolist()
    sorted_features = ["cluster", "target"] + sorted_feature_names
    data_clustering_ranked = data_clustering[sorted_features]

    
    # Sort and rank clustering dataframe 
    data_clustering_ranked = _sort_clusters_by_target(data_clustering_ranked, model_type)
    data_clustering_ranked = data_clustering_ranked.sort_values(by=["cluster", "target"])

    return feature_importance_local, feature_importance_global, data_clustering_ranked


