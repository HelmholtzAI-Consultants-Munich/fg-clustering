############################################
# imports
############################################

import sys
import numpy as np
import pandas as pd

from bisect import bisect
from scipy.stats import f_oneway, chisquare
from statsmodels.stats import multitest
from sklearn.utils import resample


############################################
# functions
############################################


def compute_balanced_average_impurity(categorical_values, cluster_labels):
    '''Compute balanced average impurity as score for categorical values in a clustering. 
    Impurity score is an Gini Coefficient of the classes within each cluster. 
    The class sizes are balanced by rescaling with the inverse size of the class in the overall dataset.

    :param categorical_values: Values of categorical feature / target.
    :type categorical_values: pandas.Series
    :param cluster_labels: Cluster labels for each value.
    :type cluster_labels: numpy.ndarray
    :return: Impurity score.
    :rtype: float
    '''
    #compute the number of datapoints for each class to use it then for rescaling of the 
    #class sizes within each cluster --> rescaling with inverse class size
    rescaling_factor = {class_: 1/sum(categorical_values==class_) for class_ in np.unique(categorical_values)} 
    balanced_impurities = []

    for cluster in np.unique(cluster_labels):
        categorical_values_cluster = categorical_values[cluster_labels == cluster]
        
        #compute balanced class probabilities (rescaled with overall class size)
        class_probabilities_unnormalized = [sum(categorical_values_cluster==class_)*rescaling_factor[class_] for class_ in np.unique(categorical_values)]
        class_probabilities_unnormalized = np.array(class_probabilities_unnormalized)
        normalization_factor = class_probabilities_unnormalized.sum()
        class_probabilities = class_probabilities_unnormalized / normalization_factor
        
        #compute (balanced) gini impurity
        gini_impurity = 1 - np.sum(class_probabilities**2)
        balanced_impurities.append(gini_impurity)
    
    score = np.mean(balanced_impurities)
    
    return score


def compute_total_within_cluster_variation(continuous_values, cluster_labels):
    '''Compute total within cluster variation as score for continuous values in a clustering.

    :param continuous_values: Values of continuous feature / target.
    :type continuous_values: pandas.Series
    :param cluster_labels: Cluster labels for each value.
    :type cluster_labels: numpy.ndarray
    :return: Within cluster variation score.
    :rtype: float
    '''
    score = 0
    for cluster in np.unique(cluster_labels):
        continuous_values_cluster = continuous_values[cluster_labels == cluster]
        score += np.var(continuous_values_cluster)*len(continuous_values_cluster)
        
    return score


def _anova_test(list_of_df):
    '''Perform one way ANOVA test on continuous features.

    :param list_of_df: List of dataframes, where each dataframe contains 
        the feature values for one cluster.
    :type list_of_df: list
    :return: P-value of ANOVA test.
    :rtype: float
    '''
    anova = f_oneway(*list_of_df)
    return anova.pvalue


def _chisquare_test(df, list_of_df):
    '''Perform chi square test on categorical features.

    :param df: Dataframe with feature and cluster.
    :type df: pandas.DataFrame
    :param list_of_df: List of dataframes, where each dataframe contains 
        the feature values for one cluster.
    :type list_of_df: list
    :return: P-value of chi square test.
    :rtype: float
    '''
    cat_vals = df.feature.unique()
    count_global = np.array([(df.feature == cat_val).sum() for cat_val in cat_vals])
    count_global = count_global / count_global.sum()

    p_values = []
    for df_ in list_of_df:
        counts_clusters = np.array([(df_ == cat_val).sum() for cat_val in cat_vals])
        number_datapoints_in_cluster = counts_clusters.sum()
        p_values.append(chisquare(counts_clusters, f_exp=count_global*number_datapoints_in_cluster).pvalue)

    _, p_values = multitest.fdrcorrection(p_values)
    return min(p_values)


def _rank_features(X, y, p_value_of_features):
    '''Rank features by lowest p-value.

    :param X: Feature matrix.
    :type X: pandas.DataFrame
    :param y: Target column.
    :type y: pandas.Series  
    :param p_value_of_features: Computed p-values of all features.
    :type p_value_of_features: dict
    :return: Ranked feature matrix.
    :rtype: pandas.DataFrame
    '''
    X_ranked = X.copy()
    X_ranked['target'] = y
    p_value_of_features['target'] = -1
    p_value_of_features['cluster'] = -1
    
    # sort features by p-value
    features_sorted = [k for k, v in sorted(p_value_of_features.items(), key=lambda item: item[1])]
    X_ranked = X_ranked.reindex(features_sorted, axis=1)

    return X_ranked


def _sort_clusters_by_target(X_ranked):
    '''Sort clusters by mean target values in clusters. 

    :param X_ranked: Filtered and ranked feature matrix.
    :type X_ranked: pandas.DataFrame
    :return: Filtered and ranked feature matrix with ordered clusters.
    :rtype: pandas.DataFrame
    '''
    means = X_ranked.groupby(['cluster']).mean().sort_values(by='target',ascending=True)
    means['target'] = range(means.shape[0])
    mapping = dict(means['target'])
    mapping = dict(sorted(mapping.items(), key=lambda item: item[1]))
    X_ranked = X_ranked.replace({'cluster': mapping})
    X_ranked['cluster'] = pd.Categorical(X_ranked['cluster'], sorted(X_ranked['cluster'].unique()))

    return X_ranked


def calculate_global_feature_importance(X, y, cluster_labels):
    '''Calculate global feature importance for each feature. 
    The higher the importance for a feature, the lower the p-value obtained by 
    an ANOVA (continuous feature) or chi-square (categorical feature) test.

    :param X: Feature matrix.
    :type X: pandas.DataFrame
    :param y: Target column.
    :type y: pandas.Series
    :param cluster_labels: Clustering labels.
    :type cluster_labels: numpy.ndarray
    :return: Feature matrix ranked by p-value of statistical test and 
        dictionary with computed p-values of all features.
    :rtype: pandas.DataFrame and dict
    ''' 
    X['cluster'] = cluster_labels
    p_value_of_features = dict()

    # statistical test for each feature
    for feature in X.columns:
        assert X[feature].astype("object").dtype != X[feature].dtype, "Feature is of type object. Please reformat to category or numeric type."

        df = pd.DataFrame({'cluster': X['cluster'], 'feature': X[feature]})
        list_of_df = [df.feature[df.cluster == cluster] for cluster in set(df.cluster)]
        
        if isinstance(X[feature].dtype, pd.api.types.CategoricalDtype):
            chisquare_p_value = _chisquare_test(df, list_of_df)
            p_value_of_features[feature] = chisquare_p_value
            
        else:  
            anova_p_value = _anova_test(list_of_df)
            p_value_of_features[feature] = anova_p_value

    X_ranked = _rank_features(X, y, p_value_of_features)
    X_ranked = _sort_clusters_by_target(X_ranked)
    X_ranked.sort_values(by=['cluster','target'], axis=0, inplace=True)
    
    return X_ranked, p_value_of_features


def _calculate_p_value_categorical(X_feature_cluster, X_feature, cluster, cluster_size, bootstraps):
    '''Calculate bootstrapped p-value for categorical features to
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
    '''
    cluster_label = [cluster]*cluster_size
    X_feature_cluster_impurity = compute_balanced_average_impurity(X_feature_cluster, cluster_label)

    bootstrapped_impurity = list()
    for b in range(bootstraps):
        bootstrapped_X_feature = resample(X_feature, replace = True, n_samples = cluster_size)
        bootstrapped_impurity.append(compute_balanced_average_impurity(bootstrapped_X_feature, cluster_label))
        
    bootstrapped_impurity = sorted(bootstrapped_impurity)
    p_value = bisect(bootstrapped_impurity, X_feature_cluster_impurity) / bootstraps
    return p_value

    
def _calculate_p_value_continuous(X_feature_cluster, X_feature, cluster_size, bootstraps):
    '''Calculate bootstrapped p-value for continuous features to
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
    '''
    X_feature_cluster_var = X_feature_cluster.var()

    bootstrapped_var = list()
    for b in range(bootstraps):
        bootstrapped_X_feature = resample(X_feature, replace = True, n_samples = cluster_size)
        bootstrapped_var.append(bootstrapped_X_feature.var())
        
    bootstrapped_var = sorted(bootstrapped_var)
    p_value = bisect(bootstrapped_var, X_feature_cluster_var) / bootstraps
    return p_value
    

def get_feature_importance_clusterwise(X, bootstraps, epsilon = sys.float_info.min):
    '''Calculate local importance of each feature within each cluster. 
    The higher the importance for a feature, the lower the variance (continuous feature) 
    or impurity (categorical feature) of that feature within the cluster.

    :param X: Feature matrix.
    :type X: pandas.DataFrame
    :param bootstraps: Number of bootstraps to be drawn for computation of p-value.
    :type bootstraps: int
    :param epsilon: Small value for log calculation to avoind log(0), defaults to sys.float_info.min
    :type epsilon: float, optional
    :return: Importance matrix with feature importance per cluster.
    :rtype: pandas.DataFrame
    '''
    X = X.copy()
    clusters = X['cluster']
    clusters_size = clusters.value_counts()
    X.drop(['cluster', 'target'], axis=1, inplace=True)

    features = X.columns.tolist()
    importance = pd.DataFrame(columns=clusters.unique(), index=features)
    
    X_categorical = X.select_dtypes(include=['category'])
    X_numeric = X.select_dtypes(exclude=['category'])

    for feature in X_categorical.columns:
        for cluster in clusters.unique():
            X_feature_cluster = X_categorical.loc[clusters == cluster, feature]
            X_feature = X_categorical[feature]
            importance.loc[feature,cluster] = 1 - (_calculate_p_value_categorical(X_feature_cluster, X_feature, cluster, clusters_size.loc[cluster], bootstraps) + epsilon)

    for feature in X_numeric.columns:
        X_numeric.loc[:,feature] = X_numeric[feature]
        for cluster in clusters.unique():
            X_feature_cluster = X_numeric.loc[clusters == cluster, feature]
            X_feature = X_numeric[feature]
            importance.loc[feature,cluster] = 1 - (_calculate_p_value_continuous(X_feature_cluster, X_feature, clusters_size.loc[cluster], bootstraps) + epsilon)
    
    return importance