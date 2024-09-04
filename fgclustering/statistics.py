############################################
# imports
############################################

import numpy as np
import pandas as pd

from bisect import bisect
from scipy.stats import f_oneway

from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.proportion import proportions_chisquare
from sklearn.utils import resample


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


def _anova_test(list_of_df):
    """Perform one way ANOVA test on continuous features.

    :param list_of_df: List of dataframes, where each dataframe contains
        the feature values for one cluster.
    :type list_of_df: list
    :return: P-value of ANOVA test.
    :rtype: float
    """
    anova = f_oneway(*list_of_df)
    return anova.pvalue


def _chisquare_test(list_of_df):
    """Perform chi square test on categorical features.

    :param df: Dataframe with feature and cluster.
    :type df: pandas.DataFrame
    :param list_of_df: List of dataframes, where each dataframe contains
        the feature values for one cluster.
    :type list_of_df: list
    :return: P-value of chi square test.
    :rtype: float
    """
    # Retrieve how many categories exist
    cat_vals = list({value for df_ in list_of_df for value in df_})

    # Observed counts for each cluster in each categories
    # Example for three clusters and two categories
    #            Cat 1, Cat 2
    # np.array([[5, 60],   # Cluster 1
    #           [20, 15],  # Cluster 2
    #           [25, 25]]) # Cluster 3
    # We add pseudocount 1 to avoid division by 0
    counts_observed = np.array(
        [
            np.array([np.sum(np.array(cluster) == category) + 1 for category in cat_vals]).tolist()
            for cluster in list_of_df
        ]
    )

    # Total counts across categories
    counts_category_total = counts_observed.sum(axis=0)

    # Expected counts under the null hypothesis of equal preferences across categories
    # We add pseudocount 1 to avoid division by 0
    counts_expected = (
        np.outer(counts_observed.sum(axis=1), counts_category_total) / counts_category_total.sum() + 1
    )

    # Perform the chi-square test
    _, p_value, _ = proportions_chisquare(counts_observed, counts_expected)

    return p_value


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

    if model_type == "classifier":
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


def calculate_global_feature_importance(X, y, cluster_labels, model_type):
    """Calculate global feature importance for each feature.
    The higher the importance for a feature, the lower the p-value obtained by
    an ANOVA (continuous feature) or chi-square (categorical feature) test.
    Returned as p-value, hence importance is 1-p-value.

    :param X: Feature matrix.
    :type X: pandas.DataFrame
    :param y: Target column.
    :type y: pandas.Series
    :param cluster_labels: Clustering labels.
    :type cluster_labels: numpy.ndarray
    :param model_type: Model type of Random Forest model: classifier or regression.
    :type model_type: str
    :return: Data Frame incl features, target and cluster numbers ranked by p-value of statistical test
        and dictionary with computed p-values of all features.
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


def _calculate_p_value_categorical(X_feature_cluster, X_feature, cluster, cluster_size, bootstraps):
    """Calculate bootstrapped p-value for categorical features to
    determine the importance of the feature for a certain cluster.
    The lower the bootstrapped p-value, the lower the impurity of the
    feature in the respective cluster.

    :param X_feature_cluster: Categorical feature values in cluster.
    :type X_feature_cluster: pandas.Series
    :param X_feature: Categorical feature values.
    :type X_feature: pandas.Series
    :param cluster: Cluster number.
    :type cluster: int
    :param cluster_size: Size of cluster, i.e. number of data points in cluster.
    :type cluster_size: int
    :param bootstraps: Number of bootstraps to be drawn for computation of p-value.
    :type bootstraps: int
    :return: Bootstrapped p-value for categorical feature.
    :rtype: float
    """
    cluster_label = [cluster] * cluster_size
    rescaling_factor = {class_: 1 for class_ in np.unique(X_feature)}
    X_feature_cluster_impurity = compute_balanced_average_impurity(
        X_feature_cluster, cluster_label, rescaling_factor=rescaling_factor
    )

    bootstrapped_impurity = list()
    for b in range(bootstraps):
        bootstrapped_X_feature = resample(X_feature, replace=False, n_samples=cluster_size)
        bootstrapped_impurity.append(
            compute_balanced_average_impurity(
                bootstrapped_X_feature, cluster_label, rescaling_factor=rescaling_factor
            )
        )

    bootstrapped_impurity = sorted(bootstrapped_impurity)
    p_value = bisect(bootstrapped_impurity, X_feature_cluster_impurity) / bootstraps
    return p_value


def _calculate_p_value_continuous(X_feature_cluster, X_feature, cluster_size, bootstraps):
    """Calculate bootstrapped p-value for continuous features to
    determine the importance of the feature for a certain cluster.
    The lower the bootstrapped p-value, the lower the variance of the
    feature in the respective cluster.

    :param X_feature_cluster: Continuous feature values in cluster.
    :type X_feature_cluster: pandas.Series
    :param X_feature: Continuous feature values.
    :type X_feature: pandas.Series
    :param cluster_size: Size of cluster, i.e. number of data points in cluster.
    :type cluster_size: int
    :param bootstraps: Number of bootstraps to be drawn for computation of p-value.
    :type bootstraps: int
    :return: Bootstrapped p-value for continuous feature.
    :rtype: float
    """
    X_feature_cluster_var = X_feature_cluster.var()

    bootstrapped_var = list()
    for b in range(bootstraps):
        bootstrapped_X_feature = resample(X_feature, replace=True, n_samples=cluster_size)
        bootstrapped_var.append(bootstrapped_X_feature.var())

    bootstrapped_var = sorted(bootstrapped_var)
    p_value = bisect(bootstrapped_var, X_feature_cluster_var) / bootstraps
    return p_value


def calculate_local_feature_importance(data_clustering_ranked, bootstraps_p_value):
    """Calculate local importance of each feature within each cluster.
    The higher the importance for a feature, the lower the variance (continuous feature)
    or impurity (categorical feature) of that feature within the cluster.
    Returned as p-value, hence importance is 1-p-value.

    :param data_clustering_ranked: Filtered and ranked data frame incl features, target and cluster numbers.
    :type data_clustering_ranked: pandas.DataFrame
    :param bootstraps_p_value: Number of bootstraps to be drawn for computation of p-value.
    :type bootstraps_p_value: int
    :return: p-value matrix of all features per cluster.
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
