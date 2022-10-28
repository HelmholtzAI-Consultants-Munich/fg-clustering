############################################
# imports
############################################

import warnings

from sklearn_extra.cluster import KMedoids

import fgclustering.utils as utils
import fgclustering.optimizer as optimizer
import fgclustering.plotting as plotting
import fgclustering.statistics as stats

import warnings
warnings.filterwarnings('ignore')

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
    :param random_state: seed for random number generator, defaults to 42
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
            self.model_type = "regression"
            print("Interpreting RandomForestRegressor")
        elif is_classifier is True:
            self.model_type = "classifier"
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


    def run(self, number_of_clusters = None, max_K = 8, method_clustering = 'pam', init_clustering = 'k-medoids++', max_iter_clustering = 100, discart_value_JI = 0.6, bootstraps_JI = 100, bootstraps_p_value = 100 , n_jobs = 1):
        '''Runs the forest-guided clustering model. The optimal number of clusters for a k-medoids clustering is computed, 
        based on the distance matrix computed from the Random Forest proximity matrix.

        :param number_of_clusters: Number of clusters for the k-medoids clustering. 
            Leave None if number of clusters should be optimized, defaults to None
        :type number_of_clusters: int, optional
        :param max_K: Maximum number of clusters for cluster score computation, defaults to 8
        :type max_K: int, optional
        :param method_clustering: Which algorithm to use. 'alternate' is faster while 'pam' is more accurate, defaults to 'pam'
        :type method_clustering: {'alternate', 'pam'}, optional
        :param init_clustering: Specify medoid initialization method. To speed up computation for large datasets use 'random'.
            See sklearn documentation for parameter description, defaults to 'k-medoids++'
        :type init_clustering: {'random', 'heuristic', 'k-medoids++', 'build'}, optional
        :param max_iter_clustering: Number of iterations for k-medoids clustering, defaults to 100
        :type max_iter_clustering: int, optional
        :param discart_value_JI: Minimum Jaccard Index for cluster stability, defaults to 0.6
        :type discart_value_JI: float, optional
        :param bootstraps_JI: Number of bootstraps to compute the Jaccard Index, defaults to 100
        :type bootstraps_JI: int, optional 
        :param bootstraps_p_value: Number of bootstraps to compute the p-value of feature importance, defaults to 100
        :type bootstraps_p_value: int, optional 
        :param n_jobs: number of jobs to run in parallel when optimizing the number of clusters. 
            n_jobs=1 means no parallel computing is used, defaults to 1
        :type n_jobs: int, optional
        '''

        if number_of_clusters is None:
            self.k = optimizer.optimizeK(self.distance_matrix, 
                                    self.y.to_numpy(), 
                                    self.model_type, 
                                    max_K, 
                                    method_clustering,
                                    init_clustering,
                                    max_iter_clustering, 
                                    discart_value_JI, 
                                    bootstraps_JI, 
                                    self.random_state,
                                    n_jobs)

            if self.k == 1:
                warnings.warn("No stable clusters were found!")
                return
            
            print(f"Optimal number of cluster is: {self.k}")
        
        else:
            self.k = number_of_clusters
            print(f"Use {self.k} as number of cluster")

        self.cluster_labels = KMedoids(n_clusters=self.k, random_state=self.random_state, init=init_clustering, method=method_clustering, max_iter=max_iter_clustering).fit(self.distance_matrix).labels_
        self._X_ranked, self.p_value_of_features = stats.calculate_global_feature_importance(self.X, self.y, self.cluster_labels)
        self._p_value_of_features_per_cluster = stats.calculate_local_feature_importance(self._X_ranked, bootstraps_p_value)


    def plot_global_feature_importance(self, save = None):
        '''Plot global feature importance based on p-values given as input, the p-values are computed using an Anova (for continuous
        variable) or a Chi-Square (for categorical variables) test. The features importance is defined by 1-p_value.
        
        :param save: Filename to save plot.
        :type save: str

        '''
        plotting._plot_global_feature_importance(self.p_value_of_features, save)


    def plot_local_feature_importance(self, thr_pvalue = 1, num_cols = 4, save = None):
        '''Plot local feature importance to show the importance of each feature for each cluster, 
        measured by variance and impurity of the feature within the cluster, i.e. the higher 
        the feature importance, the lower the feature variance / impurity within the cluster.

        :param thr_pvalue: P-value threshold for feature filtering, defaults to 1
        :type thr_pvalue: float, optional
        :param save: Filename to save plot, if None the figure is not saved, defaults to None
        :type save: str, optional
        :param num_cols: Number of plots in one row, defaults to 4.
        :type num_cols: int, optional
        '''
        # drop feature with insignificant global feature p-values
        p_value_of_features_per_cluster = self._p_value_of_features_per_cluster.copy()
        for row in p_value_of_features_per_cluster.index:
            if self.p_value_of_features[row] > thr_pvalue:
                p_value_of_features_per_cluster.drop(column, axis=0, inplace=True) 

        plotting._plot_local_feature_importance(p_value_of_features_per_cluster, thr_pvalue, num_cols, save)


    def plot_decision_paths(self, distributions = True, heatmap = True, thr_pvalue = 1, num_cols = 6, save = None):
        '''Plot decision paths of the Random Forest model.
        If distributions = True, feature distributions per cluster are plotted as boxplots (for continuous features) or barplots (for categorical features).
        If heatmap = True, feature values are plotted in a heatmap sorted by clusters.
        For both plots, features are filtered and ranked by p-values of a statistical test (ANOVA for continuous features, chi-square for categorical features).

        :param distributions: Plot feature distributions, defaults to True
        :type distributions: boolean, optional
        :param heatmap: Plot feature heatmap, defaults to True
        :type heatmap: boolean, optional
        :param thr_pvalue: P-value threshold for feature filtering, defaults to 1
        :type thr_pvalue: float, optional
        :param save: Filename to save plot, if None the figure is not saved, defaults to None
        :type save: str, optional
        :param num_cols: Number of plots in one row for the distributions plot, defaults to 6.
        :type num_cols: int, optional
        '''
        # drop insignificant values
        X_ranked = self._X_ranked.copy()
        for column in X_ranked.columns:
            if self.p_value_of_features[column] > thr_pvalue:
                X_ranked.drop(column, axis=1, inplace=True)    

        if heatmap:
            plotting._plot_heatmap(X_ranked, thr_pvalue, self.model_type, save)

        if distributions:
            plotting._plot_distributions(X_ranked, thr_pvalue, num_cols, save)

