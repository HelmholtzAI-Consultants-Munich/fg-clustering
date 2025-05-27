############################################
# imports
############################################

import numpy as np
import pandas as pd
import kmedoids
import fgclustering.utils as utils
import fgclustering.optimizer as optimizer
import fgclustering.plotting as plotting
import fgclustering.statistics as stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from typing import Union

import warnings

############################################
# Forest-guided Clustering
############################################


class FgClustering:
    """
    Forest-Guided Clustering.

    Computes feature importance based on subgroups of instances that follow similar decision
    rules within a Random Forest model. This class is designed to handle both regression and
    classification problems using a trained Random Forest model.

    :param model: Trained Random Forest model, which must be an instance of
                  `sklearn.ensemble.RandomForestRegressor` or
                  `sklearn.ensemble.RandomForestClassifier`.
    :type model: sklearn.ensemble.RandomForestRegressor or sklearn.ensemble.RandomForestClassifier
    :param data: Input data containing the feature matrix. If `target_column` is a string,
                 it should be a column name in the `data` DataFrame.
    :type data: pandas.DataFrame
    :param target_column: Name of the target column as a string, or target values as a numpy array or pd.Series.
                          If provided as a string, it must correspond to a column in the `data` DataFrame.
    :type target_column: str or numpy.ndarray or pd.Series
    :param random_state: Seed for the random number generator, used for reproducibility. Defaults to 42.
    :type random_state: int, optional
    :raises ValueError: Raised if `model` is not an instance of `sklearn.ensemble.RandomForestRegressor`
                        or `sklearn.ensemble.RandomForestClassifier`.
    """

    def __init__(
        self,
        model: Union[RandomForestClassifier, RandomForestRegressor],
        data: pd.DataFrame,
        target_column: Union[str, np.ndarray, pd.Series],
        random_state: int = 42,
    ):
        self.random_state = random_state

        # check if random forest is regressor or classifier
        if "RandomForestRegressor" in str(type(model)):
            self.model_type = "regression"
            print("Interpreting RandomForestRegressor")
        elif "RandomForestClassifier" in str(type(model)):
            self.model_type = "classification"
            print("Interpreting RandomForestClassifier")
        else:
            raise ValueError(
                f"Do not recognize {str(type(model))}. Can only work with sklearn RandomForestRegressor or RandomForestClassifier."
            )

        if type(target_column) == str:
            self.y = data.loc[:, target_column]
            self.X = data.drop(columns=[target_column])
        else:
            self.y = pd.Series(target_column)
            self.X = data

        self.y.reset_index(inplace=True, drop=True)
        self.X.reset_index(inplace=True, drop=True)

        self.proximity_matrix = utils.proximityMatrix(model, self.X)
        self.distance_matrix = 1 - self.proximity_matrix
        self.k = None
        self.cluster_labels = None

    def run(
        self,
        k: int = None,
        max_K: int = 5,
        method_clustering: str = "pam",
        init_clustering: str = "random",
        max_iter_clustering: int = 100,
        discart_value_JI: float = 0.7,
        bootstraps_JI: int = 100,
        distance_func = "wasserstein", 
        scale=True, 
        n_jobs: int = 1,
        verbose: int = 1,
    ):
        """
        Runs the forest-guided clustering model to compute the optimal number of clusters using k-medoids clustering,
        based on the distance matrix derived from the Random Forest proximity matrix. The method can either optimize
        the number of clusters or use a specified number.

        :param k: Number of clusters for the k-medoids clustering. If `None`, the number of clusters
                                    will be optimized based on the distance matrix, defaults to None.
        :type k: int, optional
        :param max_K: Maximum number of clusters to consider when computing cluster scores. Used if `k` is not provided.
                        Defaults to 8.
        :type max_K: int, optional
        :param method_clustering: Clustering algorithm to use. Options include 'fasterpam', 'fastpam1', 'pam', 'alternate',
                                'fastermsc', 'fastmsc', 'pamsil', and 'pammedsil'. Defaults to 'pam'. Use 'fasterpam'
                                for larger datasets. Refer to the k-medoids documentation for more details.
        :type method_clustering: {'fasterpam', 'fastpam1', 'pam', 'alternate', 'fastermsc', 'fastmsc', 'pamsil',
                                'pammedsil'}, optional
        :param init_clustering: Method for initializing medoids. Options include 'random', 'first', and 'build'. Defaults
                                to 'random'. See the k-medoids documentation for more information.
        :type init_clustering: {'random', 'first', 'build'}, optional
        :param max_iter_clustering: Maximum number of iterations for the k-medoids algorithm, defaults to 100.
        :type max_iter_clustering: int, optional
        :param discart_value_JI: Minimum Jaccard Index value for determining cluster stability. Clusters with Jaccard
                                Index below this value are discarded, defaults to 0.6.
        :type discart_value_JI: float, optional
        :param bootstraps_JI: Number of bootstrap iterations to compute the Jaccard Index, defaults to 100.
        :type bootstraps_JI: int, optional
        :param distance_func: Defines which distance should be calculated for feature importance. Possible values: 
                            'wasserstein', 'jensen-shannon'. Wasserstein is primarily built for continuous features, 
                            Jensen-Shannon for categorical features. 
        :type distance_func: str
        :param scale: Whether to scale numeric features by their standard deviation - only in case of Wasserstein. 
        :type scale: bool
        :param n_jobs: Number of parallel jobs to run when computing the Jaccard Index bootstraps. Defaults to 1, meaning
                    no parallel computation.
        :type n_jobs: int, optional
        :param verbose: Verbosity level for output. If set to 1, prints the optimization process including Jaccard Index
                        and scores for each number of clusters. If set to 0, no output is printed. Defaults to 1.
        :type verbose: {0, 1}, optional
        """
        if k is None:
            self.k = optimizer.optimizeK(
                distance_matrix=self.distance_matrix,
                y=self.y.to_numpy(),
                model_type=self.model_type,
                max_K=max_K,
                method_clustering=method_clustering,
                init_clustering=init_clustering,
                max_iter_clustering=max_iter_clustering,
                discart_value_JI=discart_value_JI,
                bootstraps_JI=bootstraps_JI,
                random_state=self.random_state,
                n_jobs=n_jobs,
                verbose=verbose,
            )

            if self.k == 1:
                warnings.warn("No stable clusters were found!")
                return

            print(f"Optimal number of cluster is: {self.k}")

        else:
            self.k = k
            print(f"Use {self.k} as number of cluster")

        self.cluster_labels = (
            kmedoids.KMedoids(
                n_clusters=self.k,
                method=method_clustering,
                init=init_clustering,
                metric="precomputed",
                max_iter=max_iter_clustering,
                random_state=self.random_state,
            )
            .fit(self.distance_matrix)
            .labels_
        )
        
        self.distance_of_features_per_cluster, self.distance_of_features_ranked, self.data_clustering_ranked = (
            stats.calculate_feature_importance(
                X=self.X, 
                y=self.y, 
                clusters=self.cluster_labels, 
                distance_func=distance_func, 
                model_type=self.model_type, 
                scale=scale, 
                verbose=verbose
            )
        )

    
    def calculate_statistics(self, data, target_column, distance_func = "wasserstein", scale=True, verbose=False):
        """
        Recalculates distances for each feature based on the new feature matrix, affecting all related plotting functions.
        The new feature matrix must have the same number of samples and the same ordering of samples as the original matrix.

        :param data: Input data containing the new feature matrix. If `target_column` is a string,
                     it should be a column name in the `data` DataFrame.
        :type data: pandas.DataFrame
        :param target_column: Name of the target column as a string, or target values as a numpy array or pd.Series.
                              If provided as a string, it must correspond to a column in the `data` DataFrame.
        :type target_column: str or numpy.ndarray or pd.Series
        :param distance_func: Defines which distance should be calculated for feature importance. Possible values: 
                            'wasserstein', 'jensen-shannon'. Wasserstein is primarily built for continuous features, 
                            Jensen-Shannon for categorical features. 
        :type distance_func: str
        :param scale: Whether to scale numeric features by their standard deviation - only in case of Wasserstein. 
        :type scale: bool
        """
        if type(target_column) == str:
            y = data.loc[:, target_column]
            X = data.drop(columns=[target_column])
        else:
            y = pd.Series(target_column)
            X = data

        y.reset_index(inplace=True, drop=True)
        X.reset_index(inplace=True, drop=True)
        
        self.distance_of_features_per_cluster, self.distance_of_features_ranked, self.data_clustering_ranked = (
            stats.calculate_feature_importance(
                X=X, 
                y=y, 
                clusters=self.cluster_labels, 
                distance_func=distance_func, 
                model_type=self.model_type, 
                scale=scale, 
                verbose=verbose
            )
        )

    def plot_feature_importance(
        self, thr_distance: float = 0, top_n: int = None, num_cols: int = 4, save: str = None
    ):
        """
        Plot feature importance based on global and local feature importance.
        Displays both global and local importance for top n selected features.

        :param thr_distance: Distance threshold for display. Only features with distance above this threshold. 
                            Defaults to 0 (no filtering).
        :type thr_distance: float, optional
        :param top_n: Number of top features to display in the plot. If None, all features are included.
                    Defaults to None.
        :type top_n: int, optional
        :param num_cols: Number of plots per row in the output figure. Defaults to 4.
        :type num_cols: int, optional
        :param cmap_target_dict: Dict of colours to map categorical targets
        :type cmap_target_dict: dict
        :param save: Filename to save the plot. If None, the plot will not be saved. Defaults to None.
        :type save: str, optional
        """

        # select top n features for plotting
        assert isinstance(self.distance_of_features_ranked, pd.Series), (
            f"Expected `distance_of_features_ranked` to be a Series, but got {type(self.distance_of_features_ranked)} "
            f"with shape {getattr(self.distance_of_features_ranked, 'shape', 'N/A')}."
        )

        selected_features = self.distance_of_features_ranked.index.tolist()
        if top_n:
            selected_features = selected_features[:top_n]

        plotting._plot_feature_importance(
            self.distance_of_features_ranked[selected_features],
            self.distance_of_features_per_cluster.loc[selected_features],
            thr_distance,
            top_n,
            num_cols,
            cmap_target_dict,
            save,
        )

    def plot_decision_paths(
        self,
        distributions: bool = True,
        heatmap: bool = True,
        heatmap_type: str = "static",
        thr_distance: float = 0,
        top_n: int = None,
        num_cols: int = 6,
        cmap_target_dict: dict = None,
        save: str = None,
    ):
        """
        Plot decision paths of the Random Forest model. This function generates visualizations
        to help understand feature importance and distribution across clusters.

        If `distributions` is `True`, it plots feature distributions per cluster using boxplots
        for continuous features and barplots for categorical features.

        If `heatmap` is `True`, it plots a heatmap of feature values sorted by clusters.

        Both plots filter and rank features based on feature distribution differences.
        In addition, all features or only the `top_n` features can be plotted. If `top_n`is `None` all features are plotted.

        :param distributions: Whether to plot feature distributions, defaults to `True`.
        :type distributions: bool, optional
        :param heatmap: Whether to plot the feature heatmap, defaults to `True`.
        :type heatmap: bool, optional
        :param thr_distance: Distance threshold for filtering features, defaults to `0`.
        :type thr_distance: float, optional
        :param top_n: Number of top features to retain after p-value ranking, defaults to `None` (no limit).
        :type top_n: int, optional
        :param num_cols: Number of plots per row in the distributions plot, defaults to `6`.
        :type num_cols: int, optional
        :param cmap_target_dict: Dict of colours to map categorical targets
        :type cmap_target_dict: dict
        :param save: Filename to save the plot. If `None`, the figure is not saved, defaults to `None`.
        :type save: str, optional
        """
        # drop insignificant features
        selected_features = self.distance_of_features_ranked > thr_distance
        selected_features = self.distance_of_features_ranked.index[selected_features].tolist()

        # give warning if no features selected
        if not selected_features:
            raise ValueError(
                f"No features passed the distance threshold of {thr_distance}. "
                f"Nothing to plot. Consider lowering the threshold."
            )

        # select top n features for plotting
        if top_n:
            selected_features = selected_features[:top_n]

        selected_features = ["cluster", "target"] + selected_features

        if distributions:
            plotting._plot_distributions(

                self.data_clustering_ranked[selected_features], thr_distance, top_n, num_cols, save
            )

        if heatmap:
            if self.model_type == "regression":
                plotting._plot_heatmap_regression(
                    self.data_clustering_ranked[selected_features],
                    thr_distance,
                    top_n,
                    heatmap_type,
                    save,
                )
            elif self.model_type == "classification":
                plotting._plot_heatmap_classification(
                    self.data_clustering_ranked[selected_features],
                    thr_distance,
                    top_n,
                    heatmap_type,
                    cmap_target_dict,
                    save,
                )
