# Forest Guided Clustering - Explainability for Random Forest Models

Explainability methods like variable importance pinpoint the individual contribution of each feature to the classification or regression problem, but cannot identify the role of correlated features and of feature combinations. Forest-guided clustering helps to stratify the input data instances into subgroups according to different combinations of decision rules within a Random Forest model. To compute the forest-guided clustering, k-medoids clustering is applied to a distance matrix which is computed from the Random forest proximity matrix that defines which data instances have the same set of decision rules. The optimal number of clusters k for the k-medoids clustering is determined via total within cluster varaince for regression Random Forest models, or by average balanced purity for classification Random Forest models.


## Tutorial

To optain the optimal number of clusters and plot the forest-guided clustering run the function ```forest_guided_clustering()```. For further details see the IPython Notebook ```tutorial.ipynb```. 
