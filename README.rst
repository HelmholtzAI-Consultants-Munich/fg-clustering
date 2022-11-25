|stability-stable| |Stars| |PyPI| |Docs| |Cite| 

|Open in Gitpod|

.. |stability-stable| image:: https://img.shields.io/badge/stability-stable-green.svg
.. |Stars| image:: https://img.shields.io/github/stars/HelmholtzAI-Consultants-Munich/forest_guided_clustering?logo=GitHub&color=yellow
   :target: https://github.com/HelmholtzAI-Consultants-Munich/forest_guided_clustering/stargazers
.. |PyPI| image:: https://img.shields.io/pypi/v/fgclustering.svg
   :target: https://pypi.org/project/fgclustering
.. |Docs| image:: https://readthedocs.org/projects/forest-guided-clustering/badge/?version=latest
   :target: https://forest-guided-clustering.readthedocs.io
.. |Cite| image:: https://zenodo.org/badge/397931780.svg
   :target: https://zenodo.org/badge/latestdoi/397931780
.. |Open in Gitpod| image:: https://gitpod.io/button/open-in-gitpod.svg
   :target: https://gitpod.io/#https://github.com/HelmholtzAI-Consultants-Munich/fg-clustering
   

.. raw:: html

   <div align="center">
      <a href="#readme"><img src="docs/source/FGC_Logo.png" width="200"></a>
   </div>
   

Forest-Guided Clustering - Explainability for Random Forest Models
=========================================================================

This python package is about explainability of Random Forest models. Standard explainability methods (e.g. feature importance) assume independence of model features and hence, are not suited in the presence of correlated features. The Forest-Guided Clustering algorithm does not assume independence of model features, because it computes the feature importance based on subgroups of instances that follow similar decision rules within the Random Forest model. Hence, this method is well suited for cases with high correlation among model features.

For detailed documentation and usage examples, please visit the `Read the Docs documentation <https://forest-guided-clustering.readthedocs.io/>`_. 


Installation
-------------------------------

**Requirements:**

- 3.8 <= Python < 3.11 
- :code:`pandas`, :code:`numpy`, :code:`tqdm`, :code:`numba`, :code:`numexpr`
- :code:`scikit-learn`, :code:`scipy`, :code:`statsmodels`
- :code:`matplotlib`, :code:`seaborn`

All required packages are automatically installed if installation is done via :code:`pip`.

**Install Options:**

PyPI install:

.. code:: bash

    pip install fgclustering


Note: This package depends on :code:`kmedoids` package. If you are using Windows or OSX, you may need to first install Cargo with:
:code:`curl https://sh.rustup.rs -sSf | sh`

If this does not work, please try:
.. code::
   git clone https://github.com/rust-lang/cargo
   cd cargo
   cargo build --release

For further information, please visit `this page <https://pypi.org/project/kmedoids/>`_.


Usage
-------------------------------

To get explainability of your Random Forest model via Forest-Guided Clustering, you simply need to run the following commands:

.. code:: python

   from fgclustering import FgClustering
   
   # initialize and run fgclustering object
   fgc = FgClustering(model=rf, data=data, target_column='target')
   fgc.run()
   
   # visualize results
   fgc.plot_global_feature_importance()
   fgc.plot_local_feature_importance()
   fgc.plot_decision_paths()
   
   # obtain optimal number of clusters and vector that contains the cluster label of each data point
   optimal_number_of_clusters = fgc.k
   cluster_labels = fgc.cluster_labels

where 

- :code:`model=rf` is a Random Forest Classifier or Regressor object,
- :code:`data=data` is a dataset containing the same features as required by the Random Forest model, and
- :code:`target_column='target'` is the name of the target column (i.e. *target*) in the provided dataset. 

For a detailed tutorial see the IPython Notebook :code:`tutorial.ipynb`.

Usage on big datasets
-------------------------------

If you are working with the dataset containing large number of samples, you can use one of the following strategies:

- Use the cores you have at your disposal to parallelize the optimization of the cluster number. You can do so by setting the parameter :code:`n_jobs` to a value > 1 in the :code:`run()` function.
- Use the faster implementation of the pam method that K-Medoids algorithm uses to find the clusters by setting the parameter :code:`method_clustering` to *fasterpam* in the :code:`run()` function.
- Use subsampling technique

For a detailed tutorial on the usage on big datasets, please see the Section Special Case 3 in the :code:`tutorial.ipynb`.

License
-------------------------------

The fgclustering package is MIT licensed.


Contributing
-------------------------------

Contributions are more than welcome! Everything from code to notebooks to examples and documentation are all equally valuable so please don't feel you can't contribute. To contribute please fork the project make your changes and submit a pull request. We will do our best to work through any issues with you and get your code merged into the main branch.

How to cite
-------------------------------

If Forest-Guided Clustering is useful for your research, consider citing the package:

.. code:: 

   @software{lisa_sousa_2022_6445529,
     author       = {Lisa Barros de Andrade e Sousa,
                     Helena Pelin,
                     Dominik Thalmeier,
                     Marie Piraud},
     title        = {{Forest-Guided Clustering - Explainability for Random Forest Models}},
     month        = april,
     year         = 2022,
     publisher    = {Zenodo},
     version      = {v0.2.0},
     doi          = {10.5281/zenodo.7085465},
     url          = {https://doi.org/10.5281/zenodo.7085465}
   }
