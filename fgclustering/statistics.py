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
    Impurity score is an Gini Coefficient of the classes within each cluster.
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
        dummies_all, dummies_cluster = dummies_all.align(dummies_cluster, join='outer', fill_value=0)

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

        return jensenshannon(p_ref, p_cluster), {'bins': bins}



def calculate_wasserstein_distance(X, clusters, scale=True, verbose=False):
    """
    Wrapper to calculate normalized Wasserstein distances between cluster-wise and overall feature distributions.

    :param X: Feature matrix (DataFrame or array-like).
    :param clusters: Cluster labels (array-like, same length as rows in X).
    :param scale: Whether to scale numeric features by their standard deviation.
    :param verbose: Whether to print details during processing.
    :return: DataFrame of normalized Wasserstein distances (features × clusters).
    """
    
    return calculate_feature_cluster_distances(X, clusters, wasserstein_distance_func, scale=scale, verbose=verbose)


def calculate_jensen_shannon_distance(X, clusters, verbose=False):
    """
    Wrapper to calculate normalized Jensen-Shannon distances between cluster-wise and overall feature distributions.

    :param X: Feature matrix (DataFrame or array-like).
    :param clusters: Cluster labels (array-like, same length as rows in X).
    :param verbose: Whether to print details during processing.
    :return: DataFrame of normalized Jensen-Shannon distances (features × clusters).
    """
    
    return calculate_feature_cluster_distances(X, clusters, jensen_shannon_distance_func, scale=False, verbose=verbose)


def _rank_features(data_clustering, p_value_of_features_ranked):
    """Rank features by lowest p-value.

    :param X: Feature matrix.
    :type X: pandas.DataFrame
    :param y: Target column.
    :type y: pandas.Series
    :param p_value_of_features: Computed p-values of all features.
    :type p_value_of_features: dict
    :return: Ranked feature matrix.
    :rtype: pandas.DataFrame
    """
    # Reorder columns based on sorted p-values
    sorted_features = ["cluster", "target"] + p_value_of_features_ranked.columns.tolist()
    data_clustering_ranked = data_clustering[sorted_features]

    return data_clustering_ranked


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


def calculate_local_feature_importance(data_clustering_ranked, bootstraps_p_value):
    """Calculate local importance of each feature within each cluster.
    
    :param data_clustering_ranked: Filtered and ranked data frame incl features, target and cluster numbers.
    :type data_clustering_ranked: pandas.DataFrame
    :param bootstraps_p_value: Number of bootstraps to be drawn for computation of p-value.
    :type bootstraps_p_value: int
    :return: Distance matrix of all features per cluster.
    :rtype: pandas.DataFrame
    """
    data_clustering_ranked = data_clustering_ranked.copy()
    clusters = data_clustering_ranked["cluster"]
    clusters_size = clusters.value_counts()
    data_clustering_ranked.drop(["cluster", "target"], axis=1, inplace=True)

    features = data_clustering_ranked.columns.tolist()
    p_value_of_features_per_cluster = pd.DataFrame(columns=clusters.unique(), index=features)

    for feature in data_clustering_ranked.columns:
        if isinstance(data_clustering_ranked[feature].dtype, pd.CategoricalDtype):
            for cluster in clusters.unique():
                X_feature_cluster = data_clustering_ranked.loc[clusters == cluster, feature]
                X_feature = data_clustering_ranked[feature]
                p_value_of_features_per_cluster.loc[feature, cluster] = _calculate_p_value_categorical(
                    X_feature_cluster,
                    X_feature,
                    cluster,
                    clusters_size.loc[cluster],
                    bootstraps_p_value,
                )

        elif pd.api.types.is_numeric_dtype(data_clustering_ranked[feature]):
            for cluster in clusters.unique():
                X_feature_cluster = data_clustering_ranked.loc[clusters == cluster, feature]
                X_feature = data_clustering_ranked[feature]
                p_value_of_features_per_cluster.loc[feature, cluster] = _calculate_p_value_continuous(
                    X_feature_cluster,
                    X_feature,
                    clusters_size.loc[cluster],
                    bootstraps_p_value,
                )

        else:
            raise ValueError(
                f"Feature {feature} has dytpye {data_clustering_ranked[feature].dtype} but has to be of type category or numeric!"
            )

    return p_value_of_features_per_cluster


def calculate_global_feature_importance(X, y, cluster_labels, model_type):
    """Calculate global feature importance for each feature.

    :param X: Feature matrix.
    :type X: pandas.DataFrame
    :param y: Target column.
    :type y: pandas.Series
    :param cluster_labels: Clustering labels.
    :type cluster_labels: numpy.ndarray
    :param model_type: Model type of Random Forest model: classifier or regression.
    :type model_type: str
    :return: Data Frame incl features, target and cluster numbers ranked by distance
        and dictionary with computed distance of all features.
    :rtype: pandas.DataFrame and dict
    """
    data_clustering = pd.concat([X, y.rename("target"), pd.Series(cluster_labels, name="cluster")], axis=1)
    p_value_of_features = {}

    # statistical test for each feature
    for feature in data_clustering.columns:
        if feature not in ["cluster", "target"]:
            data_feature = data_clustering[feature]

            list_of_df = [
                data_feature[data_clustering["cluster"] == cluster].to_list()
                for cluster in data_clustering["cluster"].unique()
            ]

            # Perform statistical test based on feature type
            if isinstance(data_feature.dtype, pd.CategoricalDtype):
                p_value_of_features[feature] = _chisquare_test(list_of_df)
            elif pd.api.types.is_numeric_dtype(data_feature):
                p_value_of_features[feature] = _anova_test(list_of_df)
            else:
                raise ValueError(
                    f"Feature {feature} has dytpye {data_feature.dtype} but has to be of type category or numeric!"
                )

    # Convert p-value dictionary to a DataFrame and sort by p-value
    p_value_of_features_ranked = (
        pd.DataFrame.from_dict(p_value_of_features, orient="index", columns=["p_value"])
        .T.fillna(
            1
        )  # NaN can be produced if categorical features are dummy encoded and one feature is not present in one cluster
        .sort_values(by="p_value", axis=1)
    )

    # correct p-values for multiple testing
    _, p_values_corrected = fdrcorrection(p_value_of_features_ranked.loc["p_value"].tolist())
    p_value_of_features_ranked.loc["p_value"] = p_values_corrected

    # sort and rank clustering dataframe
    data_clustering_ranked = _rank_features(data_clustering, p_value_of_features_ranked)
    data_clustering_ranked = _sort_clusters_by_target(data_clustering_ranked, model_type)
    data_clustering_ranked.sort_values(by=["cluster", "target"], axis=0, inplace=True)

    return data_clustering_ranked, p_value_of_features_ranked




