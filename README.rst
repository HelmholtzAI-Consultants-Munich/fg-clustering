|Stars| |Docs| 

.. |Stars| image:: https://img.shields.io/github/stars/HelmholtzAI-Consultants-Munich/forest_guided_clustering?logo=GitHub&color=yellow
   :target: https://github.com/HelmholtzAI-Consultants-Munich/forest_guided_clustering/stargazers
.. |Docs| image:: https://readthedocs.org/projects/forest-guided-clustering/badge/?version=latest
   :target: https://forest-guided-clustering.readthedocs.io

Forest-Guided Clustering - Explainability for Random Forest Models
=========================================================================

This python package is about explainability of Random Forest models. Standard explainability methods (e.g. feature importance) assume independence of model features and hence, are not suited in the presence of correlated features. The Forest-Guided Clustering algorithm does not assume independence of model features, because it computes the feature importance based on subgroups of instances that follow similar decision rules within the Random Forest model. Hence, this method is well suited for cases with high correlation among model features.

For detailed documentation and usage examples, please visit the `Read the Docs documentation <https://forest-guided-clustering.readthedocs.io/>`_.

Installation
-------------------------------

**Requirements:**

- Python 3.8 or greater
- :code:`pandas`, :code:`numpy`, :code:`tqdm`
- :code:`sklearn`, :code:`scikit-learn-extra`, :code:`scipy`, :code:`statsmodels`
- :code:`matplotlib`, :code:`seaborn`

All required packages are automatically installed if installation is done via :code:`pip` or :code:`conda`

**Install Options:**

To install the package run:

:code:`pip install . (Installation as python package: run inside directory)`

or if you want to develop the package:

:code:`pip install -e . (Installation as python package: run inside directory)`


Conda install (not yet working):

:code:`conda install -c conda-forge fgclustering`

PyPI install (not yet working):

:code:`pip install fgclustering`


Usage
-------------------------------

To get explainability of your Random Forest model via Forest-Guided Clustering, you simply need to run the folloiwng command:

:code:`k = forest_guided_clustering(output='fgc', data=data_boston, target_column='target', model=rf)`

where 

- :code:`output='fgc'` prefix for plot names of heatmap, boxplots and feature importance plots,
- :code:`data=data_boston` is the dataset on which the Random Forest model was trained on,
- :code:`target_column='target'` is the name of the target column (i.e. *target*) in the provided dataset and 
- :code:`model=rf` is a Random Forest Classifier or Regressor object. 

The function call returns the optimal number of clusters and visualizes the forest-guided clustering results as heatmap, boxplots and feature importance plots. For a detailed tutorial see the IPython Notebook :code:`tutorial.ipynb`.


License
-------------------------------

The fgclustering package is MIT licensed.


Contributing
-------------------------------

Contributions are more than welcome! Everything from code to notebooks to examples and documentation are all equally valuable so please don't feel you can't contribute. To contribute please fork the project make your changes and submit a pull request. We will do our best to work through any issues with you and get your code merged into the main branch.




