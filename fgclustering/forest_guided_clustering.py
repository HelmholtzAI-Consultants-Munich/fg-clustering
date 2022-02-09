############################################
# imports
############################################

from sklearn_extra.cluster import KMedoids

import fgclustering.utils as utils
import fgclustering.optimizer as optimizer
import fgclustering.plotting as plotting
import fgclustering.statistics as stats

############################################
# Forest-guided Clustering
############################################


class FgClustering():
    '''Forest-Guided Clustering. 

    Computes a feature importance based on subgroups of instances that follow similar decision 
    rules within the Random Forest model.

    :param model: Trained Random Forest model.
    :type model: sklearn.ensemble
    :param data: Input data with feature matrix. 
        If target_column is a string it has to be a column in the data.
    :type data: pandas.DataFrame
    :param target_column: Name of target column or target values as numpy array.
    :type target_column: str or numpy.ndarray
    :param random_state: [description], defaults to 42
    :type random_state: int, optional
    :raises ValueError: error raised if Random Forest model is not a 
        sklearn.ensemble.RandomForestClassifier or sklearn.ensemble.RandomForestRegressor object
    '''
    def __init__(self, model, data, target_column, random_state = 42):
        self.random_state = random_state

        # check if random forest is regressor or classifier
        is_regressor = 'RandomForestRegressor' in str(type(model))
        is_classifier = 'RandomForestClassifier' in str(type(model))
        
        if is_regressor is True:
            self.method = "regression"
            print("Interpreting RandomForestRegressor")
        elif is_classifier is True:
            self.method = "classifier"
            print("Interpreting RandomForestClassifier")
        else:
            raise ValueError(f'Do not recognize {str(type(model))}. Can only work with sklearn RandomForestRegressor or RandomForestClassifier.')

        if type(target_column)==str:
            self.y = data.loc[:,target_column]
            self.X = data.drop(columns=[target_column])
        else:
            self.y = target_column
            self.X = data
        
        self.proximity_matrix = utils.proximityMatrix(model, self.X.to_numpy())
        self.distance_matrix = 1 - self.proximity_matrix
        self.k = None
        self.cluster_labels = None

    def run(self, number_of_clusters = None, max_K = 8, bootstraps_JI = 300, max_iter_clustering = 500, discart_value_JI = 0.6):
        '''Run the forest-guided clustering model. The optimal number of clusters for a k-medoids clustering is computed, 
        based on the distance matrix computed from the Random Forest proximity matrix. Features are ranked and based on 
        statistical tests, i.e. ANOVA for continuous features and chi square for categorical features.

        :param number_of_clusters: Number of clusters for the k-medoids clustering. 
            Leave None if number of clusters should be optimized, defaults to None
        :type number_of_clusters: int, optional
        :param max_K: Maximum number of clusters for cluster score computation, defaults to 8
        :type max_K: int, optional
        :param bootstraps_JI: Number of bootstraps to compute the Jaccard Index, defaults to 300
        :type bootstraps_JI: int, optional  
        :param max_iter_clustering: Number of iterations for k-medoids clustering, defaults to 500
        :type max_iter_clustering: int, optional
        :param discart_value_JI: Minimum Jaccard Index for cluster stability, defaults to 0.6
        :type discart_value_JI: float, optional
        '''
        if number_of_clusters is None:
            self.k = optimizer.optimizeK(self.distance_matrix, 
                                    self.y.to_numpy(), 
                                    max_K, 
                                    bootstraps_JI, 
                                    max_iter_clustering, 
                                    discart_value_JI, 
                                    self.method, 
                                    self.random_state)
            print(f"Optimal number of cluster is: {self.k}")
        else:
            self.k = number_of_clusters

        self.cluster_labels = KMedoids(n_clusters=self.k, random_state=self.random_state).fit(self.distance_matrix).labels_
        self._X_ranked, self.p_value_of_features = stats.calculate_global_feature_importance(self.X, self.y, cluster_labels)

    def plot_global_feature_importance(self):
        return 0

    def plot_local_feature_importance(self, bootstraps_p_value = 1000, thr_pvalue = 0.01, save = None, num_cols = 4):
        '''Plot local feature importance to show the importance of each feature for each cluster, 
        measured by variance and impurity of the feature within the cluster, i.e. the higher 
        the feature importance, the lower the feature variance / impurity within the cluster.

        :param bootstraps_p_value: Number of bootstraps to compute the p-value of feature importance, defaults to 1000
        :type bootstraps_p_value: int, optional
        :param thr_pvalue: P-value threshold for feature filtering, defaults to 0.01
        :type thr_pvalue: float, optional
        :param save: Filename to save plot, if None the figure is not saved, defaults to None
        :type save: str, optional
        :param num_cols: Number of plots in one row, defaults to 4.
        :type num_cols: int, optional
        '''
        # drop insignificant values
        for column in X.columns:
            if self.p_value_of_features[column] > thr_pvalue:
                X.drop(column, axis  = 1, inplace=True)

        _plot_feature_importance(self.X_ranked, bootstraps_p_value, save, num_cols)

    def plot_heatmap(self, thr_pvalue = 0.01, save = None):
        '''Plot feature heatmap sorted by clusters, where features are filtered and ranked 
        with statistical tests (ANOVA for continuous featres, chi square for categorical features). 

        :param thr_pvalue: P-value threshold for feature filtering, defaults to 0.01
        :type thr_pvalue: float, optional
        :param save: Filename to save plot, if None the figure is not saved, defaults to None
        :type save: str, optional
        '''
        # drop insignificant values
        for column in X.columns:
            if self.p_value_of_features[column] > thr_pvalue:
                X.drop(column, axis  = 1, inplace=True)    

        _plot_heatmap(self.X_ranked, self.method, save)

    def plot_boxplots(self, thr_pvalue = 0.01, save = None, num_cols = 6):
        '''Plot feature boxplots divided by clusters, where features are filtered and ranked 
        with statistical tests (ANOVA for continuous featres, chi square for categorical features).

        :param thr_pvalue: P-value threshold for feature filtering, defaults to 0.01
        :type thr_pvalue: float, optional
        :param save: Filename to save plot, if None the figure is not saved, defaults to None
        :type save: str, optional
        :param num_cols: Number of plots in one row, defaults to 6.
        :type num_cols: int, optional
        '''
        # drop insignificant values
        for column in X.columns:
            if self.p_value_of_features[column] > thr_pvalue:
                X.drop(column, axis  = 1, inplace=True)

        _plot_boxplots(self.X_ranked, save, num_cols)




def fgclustering(save, data, target_column, model,   
                 max_K = 8, number_of_clusters = None, max_iter_clustering = 500,  
                 bootstraps_JI = 300, discart_value_JI = 0.6,
                 bootstraps_p_value = 1000, thr_pvalue = 0.01, random_state = 42):
    '''Run forest-guided clustering algorithm for Random Forest Classifier or Regressor. The optimal number of clusters 
    for a k-medoids clustering is computed, based on the distance matrix computed from the Random Forest proximity matrix. 
    Features are ranked and filtered based on statistical tests (ANOVA for continuous features, chi square for categorical features).
    Feature distribution per cluster is shown in a heatmap and boxplots. Feature importance is plotted to show 
    the importance of each feature for each cluster, measured by variance and impurity of the feature within the cluster, 
    i.e. the higher the feature importance, the lower the feature variance/impurity within the cluster.

    :param save: Filename to save plot.
    :type save: str
    :param data: Input data with feature matrix. 
        If target_column is a string it has to be a column in the data.
    :type data: pandas.DataFrame
    :param target_column: Name of target column or target values as numpy array.
    :type target_column: str or numpy.ndarray
    :param model: Trained Random Forest model.
    :type model: sklearn.ensemble
    :param max_K: Maximum number of clusters for cluster score computation, defaults to 6
    :type max_K: int, optional
    :param number_of_clusters: Number of clusters for the k-medoids clustering. 
        Leave None if number of clusters should be optimized, defaults to None
    :type number_of_clusters: int, optional
    :param max_iter_clustering: Number of iterations for k-medoids clustering, defaults to 500
    :type max_iter_clustering: int, optional
    :param bootstraps_JI: Number of bootstraps to compute the Jaccard Index, defaults to 300
    :type bootstraps_JI: int, optional
    :param discart_value_JI: Minimum Jaccard Index for cluster stability, defaults to 0.6
    :type discart_value_JI: float, optional
    :param bootstraps_p_value: Number of bootstraps to compute the p-value of feature importance, defaults to 10000
    :type bootstraps_p_value: int, optional
    :param thr_pvalue: P-value threshold for feature filtering, defaults to 0.05
    :type thr_pvalue: float, optional
    :param random_state: Seed number for random state, defaults to 42
    :type random_state: int, optional
    :return: Optimal number of clusters.
    :rtype: int
    '''
    # check if random forest is regressor or classifier
    is_regressor = 'RandomForestRegressor' in str(type(model))
    is_classifier = 'RandomForestClassifier' in str(type(model))
    
    if is_regressor is True:
        method = "regression"
        print("Interpreting RandomForestRegressor")
    elif is_classifier is True:
        method = "classifier"
        print("Interpreting RandomForestClassifier")
    else:
        raise ValueError(f'Do not recognize {str(type(model))}. Can only work with sklearn RandomForestRegressor or RandomForestClassifier.')
    
    if type(target_column)==str:
        y = data.loc[:,target_column]
        X = data.drop(columns=[target_column])
    else:
        y = target_column
        X = data
    
    distanceMatrix = 1 - utils.proximityMatrix(model, X.to_numpy())
    
    if number_of_clusters is None:
        k = optimizer.optimizeK(distanceMatrix, y.to_numpy(), max_K, bootstraps_JI, max_iter_clustering, discart_value_JI, method, random_state)
    else:
        k = number_of_clusters
        
    print(f"Visualizing forest guided clustering for {k} clusters")
    plotting.plot_forest_guided_clustering(save, X, y, method, distanceMatrix, k, thr_pvalue, bootstraps_p_value, random_state)
    
    return k