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

import fgclustering.utils as utils
import fgclustering.statistics as stats


############################################
# functions
############################################

def _plot_global_feature_importance(p_value_of_features, save):
    '''Plot global feature importance based on p-values given as input.

    :param p_value_of_features: dictionary where keys are names of features and values are p-values of these features
    :type p_value_of_features: dict
    :param save: Filename to save plot.
    :type save: str
    '''
    
    importance = p_value_of_features.copy()
    importance.pop('target')
    importance.pop('cluster')
    importance = pd.DataFrame(importance, index=[0])
    importance = pd.melt(importance)
    importance.sort_values(by='value', ascending=True, inplace=True)
    importance.value = 1 - importance.value

    plot = sns.barplot(data=importance, x='value', y='variable', color='darkgrey')
    plot.set_xlabel('importance')
    plot.set_ylabel('feature')
    plt.title('Global Feature Importance')
    plt.tight_layout()

    if save is not None:
        plt.savefig('{}_global_feature_importance.png'.format(save), bbox_inches='tight', dpi = 300)
    plt.show()
    

def _plot_local_feature_importance(X, bootstraps, save, num_cols):
    '''Plot local feature importance to show the importance of each feature for each cluster. 

    :param X: Feature matrix.
    :type X: pandas.DataFrame
    :param bootstraps: Number of bootstraps to be drawn for computation of p-value.
    :type bootstraps: int
    :param save: Filename to save plot.
    :type save: str
    :param num_cols: Number of plots in one row.
    :type num_cols: int
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

    if save is not None:
        plt.savefig('{}_local_feature_importance.png'.format(save), bbox_inches='tight', dpi = 300)
    plt.show()
    

def _plot_heatmap(X, method, save):
    '''Plot feature heatmap sorted by clusters, where features are filtered and ranked 
    with statistical tests (ANOVA for continuous featres, chi square for categorical features). 

    :param X: Feature matrix.
    :type X: pandas.DataFrame
    :param method: Model type of Random Forest model: classifier or regression.
    :type method: str
    :param save: Filename to save plot.
    :type save: str
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
    
    if save is not None:
        plt.savefig('{}_heatmap.png'.format(save), bbox_inches='tight', dpi = 300)
    plt.show()
    

def _plot_boxplots(X, save, num_cols):
    '''Plot feature boxplots divided by clusters, where features are filtered and ranked 
    with statistical tests (ANOVA for continuous featres, chi square for categorical features).

    :param X: Feature matrix.
    :type X: pandas.DataFrame
    :param save: Filename to save plot.
    :type save: str
    :param num_cols: Number of plots in one row.
    :type num_cols: int
    '''    
    
    target_and_features = X.columns[X.columns != 'cluster']
    X_boxplot = pd.melt(X, id_vars=['cluster'], value_vars=target_and_features)
    
    plot = sns.FacetGrid(X_boxplot, col='variable', height=3, sharey=False, col_wrap=num_cols)
    plot.map(sns.boxplot, 'cluster', 'value', color='darkgrey')
    plot.set_axis_labels('Cluster', 'Feature Value', clear_inner=False)
    plot.set_titles(col_template="Feature: {col_name}")

    if save is not None:
        plt.savefig('{}_boxplots.png'.format(save), bbox_inches='tight', dpi = 300)
    plt.show()
        
