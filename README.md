# Forest-Guided Clustering - Explainability for Random Forest Models

This python package is about explainability of Random Forest models. Since standard methods are not suited in the presence of correlated features, the Forest-Guided Clustering package computes feature importance based on subgroups of instances that follow similar decision rules and returns the optimal value for k.

For detailed documentation and usage examples, please visit the [documentation](https://forest-guided-clustering.readthedocs.io/en/latest/index.html).

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

where 

- ```output='fgc'``` sets the name for the heatmap and boxplot,
- ```model=rf``` is a Random Forest Classifier or Regressor object, 
- ```data=data_boston``` is the dataset on which the Random Forest model was trained on and 
- ```target_column='target'``` is the name of the target column in the provided dataset. 

The function will return the optimal number of clusters and plot the forest-guided clustering results as heatmap and boxplots.

For a detailed tutorial see the IPython Notebook ```tutorial.ipynb```.
