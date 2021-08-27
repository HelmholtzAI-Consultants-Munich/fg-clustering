############################################
# imports
############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import f_oneway
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


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
    X_anova.loc['p_value','cluster'] = 0  
    X_anova = X_anova.transpose()
    X_anova = X_anova.loc[X_anova.p_value < thr_pvalue]
    X_anova.sort_values(by='p_value', inplace=True)
    X_anova.drop('p_value', axis=1, inplace=True)
    X_anova.sort_values(by=['cluster','target'], axis=1, inplace=True)
    
    return X_anova.transpose()
    
    
    
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
    
    plot = sns.heatmap(X_heatmap.transpose(), xticklabels=False, yticklabels = 1, cmap='coolwarm', cbar_kws={'label': 'standardized feature values'})
    plot.set(title='Forest-Guided Clustering')
    plot.set_yticklabels(X_heatmap.columns, size = 6)
    plt.savefig(output, bbox_inches='tight', dpi = 300)
    #plt.close() 
    

    
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