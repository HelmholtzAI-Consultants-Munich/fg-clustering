############################################
# imports
############################################

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn_extra.cluster import KMedoids

import src.utils as utils
import src.statistics as stats


############################################
# Plot forest-guided clustering results as heatmap
############################################

def _plot_heatmap(output, X, method):
    '''Plot feature heatmap sorted by clusters, where features are filtered and ranked 
    with statistical tests (ANOVA for continuous featres, chi square for categorical features). 

    :param output: Filename to save plot.
    :type output: str
    :param X: Feature matrix.
    :type X: pandas.DataFrame
    :param method: Model type of Random Forest model: classifier or regression.
    :type method: str
    '''
    X_scaled = utils.scale_minmax(X)
    X_heatmap = pd.DataFrame(columns = X_scaled.columns)

    target_values_original = X['target']
    target_values_scaled = X_scaled['target']
    
    one_percent_of_number_of_samples = int(np.ceil(0.01*len(X)))

    for cluster in X_scaled.cluster.unique():
        X_heatmap = X_heatmap.append(X_scaled[X_scaled.cluster == cluster], ignore_index=True)
        X_heatmap = X_heatmap.append(pd.DataFrame(np.nan, 
                                                  index = np.arange(one_percent_of_number_of_samples), #blank lines which are 1% of num samples
                                                  columns = X_scaled.columns), ignore_index=True)
    X_heatmap = X_heatmap[:-5]
    X_heatmap.drop('cluster', axis=1, inplace=True)

    n_samples, n_features = X_heatmap.shape
    heatmap_ = np.zeros((n_features,n_samples,4))

    cmap_features = matplotlib.cm.get_cmap('coolwarm').copy()
    cmap_target = matplotlib.cm.get_cmap('viridis') #gist_ncar

    for feature in range(n_features):
        for sample in range(n_samples):
            if feature == 0:
                heatmap_[feature,sample,:] = cmap_target(X_heatmap.iloc[sample, feature])
            else:
                heatmap_[feature,sample,:] = cmap_features(X_heatmap.iloc[sample, feature])

    fig = plt.figure()
    img = plt.imshow(heatmap_, interpolation='none', aspect='auto')

    plt.title('Forest-Guided Clustering')
    plt.xticks([], [])
    plt.yticks(range(n_features), X_heatmap.columns)

    # remove bounding box
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    if method == "regression":
        norm = matplotlib.colors.Normalize(vmin=target_values_original.min(), vmax=target_values_original.max())
        cbar_target = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_target))
        cbar_target.set_label('target')
    else:
        legend_elements = [Patch(facecolor=cmap_target(tv_n), 
                                 edgecolor=cmap_target(tv_n),
                                 label=f'{tv_o}') for tv_n, tv_o in zip(target_values_scaled.unique(), target_values_original.unique())]
        ll = plt.legend(handles=legend_elements,
                        bbox_to_anchor=(0, 0), 
                        loc='upper left', 
                        ncol=min(6,len(legend_elements)),
                        title="target") 
    
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cbar_features = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_features))
    cbar_features.set_label('standardized feature values')
    
    plt.savefig('{}_heatmap.png'.format(output), bbox_inches='tight', dpi = 300)
    plt.show()
    

def _plot_boxplots(output, X, num_cols = 6):
    '''Plot feature boxplots divided by clusters, where features are filtered and ranked 
    with statistical tests (ANOVA for continuous featres, chi square for categorical features).

    :param output: Filename to save plot.
    :type output: str
    :param X: Feature matrix.
    :type X: pandas.DataFrame
    :param num_cols: Number of plots in one row, defaults to 6.
    :type num_cols: int, optional
    '''    
    target_and_features = X.columns[X.columns != 'cluster']
    X_boxplot = pd.melt(X, id_vars=['cluster'], value_vars=target_and_features)
    
    plot = sns.FacetGrid(X_boxplot, col='variable', height=3, sharey=False, col_wrap=num_cols)
    plot.map(sns.boxplot, 'cluster', 'value', color='darkgrey')
    plot.set_axis_labels('Cluster', 'Feature Value')
    plot.set_titles(col_template="Feature: {col_name}")
    plt.savefig('{}_boxplots.png'.format(output), bbox_inches='tight', dpi = 300)
    plt.show()
    

def _plot_feature_importance(output, X, bootstraps, num_cols = 4):
    '''Plot feature importance to show the importance of each feature for each cluster, 
    measured by variance and impurity of the feature within the cluster, i.e. the higher 
    the feature importance, the lower the feature variance / impurity within the cluster.

    :param output: Filename to save plot.
    :type output: str
    :param X: Feature matrix.
    :type X: pandas.DataFrame
    :param num_cols: Number of plots in one row, defaults to 4.
    :type num_cols: int, optional
    '''
    importance = stats.get_feature_importance_clusterwise(X, bootstraps)
    num_features = len(importance)

    X_barplot = pd.melt(importance , ignore_index=False)
    X_barplot = X_barplot.rename_axis('feature').reset_index(level=0, inplace=False)
    X_barplot = X_barplot.sort_values('value',ascending=False)

    height = max(5,int(np.ceil(5*num_features/25)))
    num_cols = min(num_cols, len(importance.columns))

    plot = sns.FacetGrid(X_barplot, col='variable', sharey=False, col_wrap=num_cols,height=height)
    plot.map(sns.barplot, 'value', 'feature', color='darkgrey')
    plot.set_axis_labels('importance', 'feature')
    plot.set_titles(col_template="Cluster {col_name}")
    plt.suptitle('Feature Importance per Cluster')
    plt.tight_layout()

    plt.savefig('{}_feature_importance.png'.format(output), bbox_inches='tight', dpi = 300)
    plt.show()
        

def plot_forest_guided_clustering(output, X, y, method, distanceMatrix, k, thr_pvalue, bootstraps, random_state):
    '''Plot results of forest-guided clustering. Rank and filter feature matrix based on staistical tests
    (ANOVA for continuous featres, chi square for categorical features). Show feature distribution per cluster
    in heatmap and boxplot. Plot feature importance to show the importance of each feature for each cluster, 
    measured by variance and impurity of the feature within the cluster, i.e. the higher the feature importance,
    the lower the feature variance / impurity within the cluster.

    :param output: Filename to save plot.
    :type output: str
    :param X: Feature matrix.
    :type X: pandas.DataFrame
    :param y: Target column.
    :type y: pandas.Series
    :param method: Model type of Random Forest model: classifier or regression.
    :type method: str
    :param distanceMatrix: Distance matrix computed from Random Forest proximity matrix.
    :type distanceMatrix: pandas.DataFrame
    :param k: Number of cluster for k-medoids clustering.
    :type k: int
    :param thr_pvalue: P-value threshold for feature filtering.
    :type thr_pvalue: float
    :param bootstraps: Number of bootstraps to be drawn for computation of p-value.
    :type bootstraps: int, optional
    :param random_state: Seed number for random state.
    :type random_state: int
    '''
    cluster_labels = KMedoids(n_clusters=k, random_state=random_state).fit(distanceMatrix).labels_
    X_ranked = stats.feature_ranking(X, y, cluster_labels, thr_pvalue)
    
    _plot_heatmap(output, X_ranked, method)
    _plot_boxplots(output, X_ranked)
    _plot_feature_importance(output, X_ranked, bootstraps)

