Basic Usage
-------------------------------

To get explainability of your Random Forest model via Forest-Guided Clustering, you simply need to run the folloiwng command:

.. code:: python

   from forest_guided_clustering import fgclustering
   k_opt = fgclustering(output='fgc', data=data_boston, target_column='target', model=rf)

where 

- :code:`output='fgc'` prefix for plot names of heatmap, boxplots and feature importance plots,
- :code:`data=data_boston` is the dataset on which the Random Forest model was trained on,
- :code:`target_column='target'` is the name of the target column (i.e. *target*) in the provided dataset and 
- :code:`model=rf` is a Random Forest Classifier or Regressor object. 

The function call returns the optimal number of clusters :code:`k_opt` and visualizes the forest-guided clustering results as heatmap, boxplots and feature importance plots. For a detailed tutorial see the IPython Notebook :code:`tutorial.ipynb`.
