############################################
# imports
############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import f_oneway, chisquare
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from statsmodels.stats import multitest


############################################
# Plot forest-guided clustering results as heatmap
############################################


def _scale_standard(X):
    """
    Feature Scaling with StandardScaler. 
    Parameters
    ----------
        X: Pandas DataFrame
            Feature matrix.

    Returns
    -------
        X_scale: Pandas DataFrame
            Scaled feature matrix.
    """
    
    SCALE = StandardScaler()
    SCALE.fit(X)

    X_scale = pd.DataFrame(SCALE.transform(X))
    X_scale.columns = X.columns
    X_scale.reset_index(inplace=True,drop=True)

    return X_scale



def _scale_minmax(X):
    """
    Feature Scaling with MinMaxScaler. 
    Parameters
    ----------
        X: Pandas DataFrame
            Feature matrix.

    Returns
    -------
        X_scale: Pandas DataFrame
            Scaled feature matrix.
    """
        
    SCALE = MinMaxScaler()
    SCALE.fit(X)

    X_scale = pd.DataFrame(SCALE.transform(X))
    X_scale.columns = X.columns
    X_scale.reset_index(inplace=True,drop=True)

    return X_scale


def _sort_clusters_by_target(X_anova):
    means = X_anova.groupby(['cluster']).mean().sort_values(by='target',ascending=True)
    means['target'] = range(means.shape[0])
    mapping = dict(means['target'])
    mapping = dict(sorted(mapping.items(), key=lambda item: item[1]))
    X_anova['cluster'] = pd.Categorical(X_anova['cluster'], list(mapping.keys()))

    return X_anova


def _anova_test(X, y, cluster_labels, thr_pvalue):
    """
    Selecting significantly different features across clusters with an ANOVA test. 
    Parameters
    ----------
        X: Pandas DataFrame
            Feature matrix.
        y: Pandas Series
            Target column.
        cluster_labels: numpy array
            Clustering labels.
        thr_pvalue: int
            P-value threshold for feature filtering. 

    Returns
    -------
        X_anova: Pandas DataFrame
            Feature matrix filtered by ANOVA p-value.
    """
    
    X['cluster'] = cluster_labels
    X_test = X.copy()
    X_test['target'] = y
    p_value_of_features = dict()

    # anova test
    for feature in X.columns:
        assert X[feature].astype("object").dtype != X[feature].dtype, "Feature is of type object. Please reformat to category or numeric type."

        df = pd.DataFrame({'cluster': X['cluster'], 'feature': X[feature]})
        list_of_df = [df.feature[df.cluster == cluster] for cluster in set(df.cluster)]
        
        if isinstance(X_test[feature].dtype, pd.api.types.CategoricalDtype):
            cat_vals = df.feature.unique()

            count_global = np.array([(df.feature == cat_val).sum() for cat_val in cat_vals])
            count_global = count_global / count_global.sum()

            p_values = []
            for df_ in list_of_df:
                counts_clusters = np.array([(df_ == cat_val).sum() for cat_val in cat_vals])
                number_datapoints_in_cluster = counts_clusters.sum()
                p_values.append(chisquare(counts_clusters, f_exp=count_global*number_datapoints_in_cluster).pvalue)

            _, p_values = multitest.fdrcorrection(p_values)
            p_value_of_features[feature] = min(p_values)
            
        else:        
            anova = f_oneway(*list_of_df)
            p_value_of_features[feature] = anova.pvalue

    p_value_of_features['target'] = -1
    p_value_of_features['cluster'] = -1
    
    # sort features by p-value
    features_sorted = [k for k, v in sorted(p_value_of_features.items(), key=lambda item: item[1])]
    X_test = X_test.reindex(features_sorted, axis=1)
    
    # drop insignificant values
    for column in X_test.columns:
        if p_value_of_features[column] > thr_pvalue:
            X_test = X_test.drop(column, axis  = 1)
            
    X_test = _sort_clusters_by_target(X_test)
    X_test.sort_values(by=['cluster','target'], axis=0, inplace=True)
    
    return X_test
    
    
    
def _plot_heatmap(output, X_anova):
    """
    Plot heatmap with significant features sorted by clusters. 
    Parameters
    ----------
        output: string
            Filename for heatmap plot.
        X_anova: Pandas DataFrame
            Feature matrix filtered by ANOVA p-value.

    Returns
    -------
        --- None ---
    """
    
    X_anova = _scale_minmax(X_anova)

    X_heatmap = pd.DataFrame(columns = X_anova.columns)
    for cluster in X_anova.cluster.unique():
        X_heatmap = X_heatmap.append(X_anova[X_anova.cluster == cluster], ignore_index=True)
        X_heatmap = X_heatmap.append(pd.DataFrame(np.nan, index = np.arange(5), columns = X_anova.columns), ignore_index=True)
    X_heatmap = X_heatmap[:-5]
    X_heatmap.drop('cluster', axis=1, inplace=True)
    
    plot = sns.heatmap(X_heatmap.transpose(), xticklabels=False, yticklabels = 1, cmap='coolwarm', cbar_kws={'label': 'standardized feature values'})
    plot.set(title='Forest-Guided Clustering')
    plot.set_yticklabels(X_heatmap.columns, size = 6)
    plt.savefig('{}_heatmap.png'.format(output), bbox_inches='tight', dpi = 300)
    plt.show()
    
    
    
def _plot_boxplots(output, X_anova, num_cols = 6):
    """
    Plot boxplots of significant features devided by clusters. 
    Parameters
    ----------
        output: string
            Filename for heatmap plot.
        X_anova: Pandas DataFrame
            Feature matrix filtered by ANOVA p-value.

    Returns
    -------
        --- None ---
    """
    
    target_and_features = X_anova.columns[X_anova.columns != 'cluster']
    X_boxplot = pd.melt(X_anova, id_vars=['cluster'], value_vars=target_and_features)
    
    plot = sns.FacetGrid(X_boxplot, col='variable', height=3, sharey=False, col_wrap=num_cols)
    plot.map(sns.boxplot, 'cluster', 'value')
    plt.savefig('{}_boxplots.png'.format(output), bbox_inches='tight', dpi = 300)
    plt.show()
    
    

def _feature_importance_clusterwise(X_anova):
    
    for feature in X_anova.columns:
        if feature not in ['target','cluster']:
            print(feature)
            if isinstance(X_anova[feature].dtype, pd.api.types.CategoricalDtype):
                print('True')
            else:
                print('False')

        
        
    
    
def plot_forest_guided_clustering(output, model, distanceMatrix, data, target_column, k, thr_pvalue, random_state):
    """
    Plot forest-guided clustering results as heatmap but exclude feature that show no significant difference across clusters. 
    Parameters
    ----------
        output: string
            Filename for heatmap plot.
        model: sklearn object
            Trained Random Forest model.
        distanceMatrix: Pandas DataFrame
            Distance matrix computed from Random Forest proximity matrix.
        data: Pandas DataFrame
            Input data with feature matrix and target column.
        target_column: string
            Name of target column.
        k: int
            Number of cluster for k-medoids clustering.
        thr_pvalue: int
            P-value threshold for feature filtering.
        random_state: int
            Seed number for random state.

    Returns
    -------
        --- None ---
    """
    
    X = data.loc[:, data.columns != target_column]
    y = data[target_column]
    cluster_labels = KMedoids(n_clusters=k, random_state=random_state).fit(distanceMatrix).labels_
    
    X_anova = _anova_test(X, y, cluster_labels, thr_pvalue)
    
    _plot_heatmap(output, X_anova)
    _plot_boxplots(output, X_anova)