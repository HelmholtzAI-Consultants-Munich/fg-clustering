.. Forest Guided Clustering documentation master file, created by
   sphinx-quickstart on Thu Dec  9 16:39:21 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Forest-Guided Clustering - Explainability for Random Forest Models
=========================================================================

This python package is about explainability of Random Forest models. Standard explainability methods (e.g. feature importance) assume independence of model features and hence, are not suited in the presence of correlated features. The Forest-Guided Clustering algorithm does not assume independence of model features, because it computes the feature importance based on subgroups of instances that follow similar decision rules within the Random Forest model. Hence, this method is well suited for cases with high correlation among model features.

**Citation:** If Forest-Guided Clustering is useful for your research, consider citing the package via `DOI: 10.5281/zenodo.7085465 <https://zenodo.org/badge/latestdoi/397931780>`_.

.. image:: ./FGC_workflow.png

Quick Start
==================

**Installation:**

PyPI install:

.. code:: bash

    pip install fgclustering


*Note:* This package depends on :code:`kmedoids` package. If you are using Windows or OSX, you may need to first install Cargo with:

.. code:: bash 
   
   curl https://sh.rustup.rs -sSf | sh

If this does not work, please try to install Cargo from source:

.. code:: bash

   git clone https://github.com/rust-lang/cargo
   cd cargo
   cargo build --release


For further information, please visit `this page <https://pypi.org/project/kmedoids/>`_.


**Basic Usage:**

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


**Usage on big datasets**

If you are working with the dataset containing large number of samples, you can use one of the following strategies:

- Use the cores you have at your disposal to parallelize the optimization of the cluster number. You can do so by setting the parameter :code:`n_jobs` to a value > 1 in the :code:`run()` function.
- Use the faster implementation of the pam method that K-Medoids algorithm uses to find the clusters by setting the parameter :code:`method_clustering` to *fasterpam* in the :code:`run()` function.
- Use subsampling technique

For a detailed tutorial on the usage on big datasets, please see the Section Special Case 3 in the :code:`tutorial.ipynb`.


Table of Content
==================

.. toctree::
   :maxdepth: 2
   
   introduction
   general_algorithm
   feature_importance
   tutorial

**API Reference:**

.. toctree::
   :maxdepth: 1
   
   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
