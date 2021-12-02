############################################
# imports
############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from bisect import bisect
from scipy.stats import f_oneway
from sklearn.utils import resample

from scipy.stats import f_oneway, chisquare
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import src.optimizer as opt
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
    
    
def _plot_heatmap(output, X_anova, method):
    """
    Plot heatmap with significant features sorted by clusters. 
    Parameters
    ----------
        output: string
            Filename for heatmap plot.
        X_anova: Pandas DataFrame
            Feature matrix filtered by ANOVA p-value.
        method: string
            Model type of Random Forest model: classifier or regression.

    Returns
    -------
        --- None ---
    """
    target_values_original = X_anova['target']


    X_anova = _scale_minmax(X_anova)

    target_values_normalized = X_anova['target']

    number_of_samples = len(X_anova)
    one_percent_of_number_of_samples = int(np.ceil(0.01*number_of_samples))

    X_heatmap = pd.DataFrame(columns = X_anova.columns)
    for cluster in X_anova.cluster.unique():
        X_heatmap = X_heatmap.append(X_anova[X_anova.cluster == cluster], ignore_index=True)
        X_heatmap = X_heatmap.append(pd.DataFrame(np.nan, 
                                                  index = np.arange(one_percent_of_number_of_samples), #blank lines which are 1% of num samples
                                                  columns = X_anova.columns), 
                                     ignore_index=True)
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

    plt.xticks([], [])
    plt.yticks(range(n_features), X_heatmap.columns)

    # remove bounding box
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.title('Forest-Guided Clustering')

    if method == "regression":
        norm = matplotlib.colors.Normalize(vmin=target_values_original.min(), vmax=target_values_original.max())

        cbar_target = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_target))
        cbar_target.set_label('target')
        
    else:
        legend_elements = [Patch(facecolor=cmap_target(tv_n), edgecolor=cmap_target(tv_n),
                                 label=f'{tv_o}') for tv_n, tv_o in zip(target_values_normalized.unique(), target_values_original.unique())]

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

    X_anova = X_anova.copy()
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
        


def plot_forest_guided_clustering(output, model, distanceMatrix, data, target_column, k, thr_pvalue, random_state, method):
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
        method: string
            Model type of Random Forest model: classifier or regression.

    Returns
    -------
        --- None ---
    """
    
    X = data.loc[:, data.columns != target_column]
    y = data[target_column]
    cluster_labels = KMedoids(n_clusters=k, random_state=random_state).fit(distanceMatrix).labels_
    
    X_anova = _anova_test(X, y, cluster_labels, thr_pvalue)
    
    _plot_heatmap(output, X_anova, method)
    _plot_boxplots(output, X_anova)
    _plot_feature_importance(output, X_anova)

