############################################
# imports
############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from bisect import bisect
from scipy.stats import f_oneway
from sklearn.utils import resample
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import src.optimizer as opt

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
    X_anova = X.copy()
    X_anova['target'] = y
    X_anova.loc['p_value'] = None

    # anova test
    for feature in X.columns:
        df = pd.DataFrame({'cluster': X['cluster'], 'feature': X[feature]})
        list_of_df = [df.feature[df.cluster == cluster] for cluster in set(df.cluster)]
        anova = f_oneway(*list_of_df)
        X_anova.loc['p_value',feature] = anova.pvalue

    X_anova.loc['p_value','target'] = -1
    X_anova.loc['p_value','cluster'] = -1  
    X_anova = X_anova.transpose()
    X_anova = X_anova.loc[X_anova.p_value < thr_pvalue]
    
    X_anova.sort_values(by='p_value', axis=0, inplace=True)
    X_anova.drop('p_value', axis=1, inplace=True)
    
    X_anova = X_anova.transpose()
    X_anova = _sort_clusters_by_target(X_anova)
    X_anova.sort_values(by=['cluster','target'], axis=0, inplace=True)
    
    return X_anova
    
    
    
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
    plot.map(sns.boxplot, 'cluster', 'value', color='darkgrey')
    plot.set_axis_labels('Cluster', 'Feature Value')
    plot.set_titles(col_template="Feature: {col_name}")
    plt.savefig('{}_boxplots.png'.format(output), bbox_inches='tight', dpi = 300)
    plt.show()
    

def _calculate_p_value_categorical(y, y_all, cluster, cluster_size, bootstraps = 1000):

    labels = [cluster]*cluster_size
    y_impurity = opt.compute_balanced_average_impurity(y, labels)

    bootstrapped_impurity = list()
    for b in range(bootstraps):
        bootstrapped_y = resample(y_all, replace = True, n_samples = cluster_size)
        bootstrapped_impurity.append(opt.compute_balanced_average_impurity(bootstrapped_y, labels))
        
    bootstrapped_impurity = sorted(bootstrapped_impurity)
    p_value = (bisect(bootstrapped_impurity, y_impurity)+1) / (bootstraps+1)
    return p_value

    
def _calculate_p_value_continuous(y, y_all, cluster_size, bootstraps = 1000):
    
    bootstrap_samples = list()
    for b in range(bootstraps):
        sample = resample(y_all, replace = True, n_samples = cluster_size)
        bootstrap_samples.append(sample.var())
        
    bootstrap_samples = sorted(bootstrap_samples)
    p_value = (bisect(bootstrap_samples, y.var())+1) / (bootstraps+1)
    return p_value
    

def _get_feature_importance_clusterwise(X_anova):
    
    X_anova.loc[:,'CHAS'] = X_anova['CHAS'].astype('category')
    
    clusters = X_anova['cluster']
    clusters_size = clusters.value_counts()
    X_anova.drop('cluster', axis=1, inplace=True)
    X_anova.drop('target', axis=1, inplace=True)
    
    X_categorical = X_anova.select_dtypes(include=['category'])
    X_numeric = X_anova.select_dtypes(exclude=['category'])
    
    features = X_anova.columns.tolist()
    var_tot = X_numeric.to_numpy().flatten().var()
    
    importance = pd.DataFrame(columns=clusters.unique(), index=features)

    for feature in X_categorical.columns:
        for cluster in clusters.unique():
            y = X_categorical.loc[clusters == cluster, feature]
            y_all = X_categorical[feature]
            importance.loc[feature,cluster] = - np.log(_calculate_p_value_categorical(y, y_all, cluster, clusters_size.loc[cluster]))

    for feature in X_numeric.columns:
        X_numeric.loc[:,feature] = X_numeric[feature] / var_tot # normalize by total variance 
        for cluster in clusters.unique():
            y = X_numeric.loc[clusters == cluster, feature]
            y_all = X_numeric[feature]
            importance.loc[feature,cluster] = - np.log(_calculate_p_value_continuous(y, y_all, clusters_size.loc[cluster]))

    return importance
            

def _plot_feature_importance(output, X_anova, num_cols = 6):

    num_clusters = len(X_anova['cluster'].unique())
    importance = _get_feature_importance_clusterwise(X_anova)

    num_rows = int(num_clusters / num_cols) + (num_clusters % num_cols > 0)
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.suptitle('Feature Importance per Cluster')

    for i in range(num_clusters):
        cluster = importance.columns[i]
        importance.sort_values(by=[cluster], inplace = True)
        X_plot = pd.DataFrame({'cluster': [cluster]*importance.shape[0], 'feature': importance.index, 'importance': importance[cluster]})

        ax = fig.add_subplot(num_rows, num_cols, i+1)
        #col = int(i/ num_cols)
        #row = i - (num_cols * col)
        
        sns.barplot(ax=ax, data=X_plot, x='importance', y='feature', order=X_plot.sort_values('importance',ascending = False).feature, color='darkgrey')

    #fig.tight_layout()
    plt.savefig('{}_feature_importance.png'.format(output), bbox_inches='tight', dpi = 300)
    plt.show()
        

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

    
    #_plot_heatmap(output, X_anova)
    #_plot_boxplots(output, X_anova)
    _plot_feature_importance(output, X_anova.copy())