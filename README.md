# Forest-Guided Clustering - Explainability for Random Forest Models

This python package is about explainability of Random Forest models. Since standard methods are not suited in the presence of correlated features, the Forest-Guided Clustering package computes feature importance based on subgroups of instances that follow similar decision rules and returns the optimal value for k.


## Description

Explainability methods like variable importance pinpoint the individual contribution of each feature to the classification or regression problem, but cannot identify the role of correlated features and of feature combinations. Forest-Guided Clustering helps to stratify the input data instances into subgroups according to different combinations of decision rules within a Random Forest model. To compute the Forest-Guided Clustering, k-medoids clustering is applied to a distance matrix, which is computed from the Random Forest proximity matrix that indicates which data instances follow the same set of decision rules. The optimal number of clusters k for the k-medoids clustering is determined via total within cluster varaince for regression Random Forest models, or by average balanced purity for classification Random Forest models.


![Forest-guided clustering approach.](./data/fgc_overview)   
**Pipeline of the Forest-Guided Clustering explainability method**

## Installation

The code has been implemented using Python 3.8. To install the package run:

```
pip install .        (Installation as python package: run inside directory)
``` 
or if you want to develop the package:
```
pip install -e .        (Installation as python package: run inside directory)
``` 


## Usage

To get explainability of your Random Forest model via Forest-Guided Clustering, you simply need to run the folloiwng command:

```
forest_guided_clustering(output='fgc', model=rf, data=data_boston, target_column='target')
```

where ```output='fgc'``` sets the name for the heatmap and boxplot, ```model=rf``` is a Random Forest Classifier or Regressor object, ```data=data_boston``` is the dataset on which the Random Forest model was trained on and ```target_column='target'``` is the name of the target column in the provided dataset. The function will return the optimal number of clusters and plot the forest-guided clustering results as heatmap and boxplots.

For a detailed tutorial see the IPython Notebook ```tutorial.ipynb```. 
