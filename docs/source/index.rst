.. Forest Guided Clustering documentation master file, created by
   sphinx-quickstart on Thu Dec  9 16:39:21 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to *Forest-Guided Clustering*'s documentation
=========================================================================

This python package is about explainability of Random Forest models. Standard explainability methods (e.g. feature importance) assume independence of model features and hence, 
are not suited in the presence of correlated features. The Forest-Guided Clustering algorithm does not assume independence of model features, because it computes the feature 
importance based on subgroups of instances that follow similar decision rules within the Random Forest model. Hence, this method is well suited for cases with high correlation among model features. 
To showcase the advantages of FGC over Feature Importance we applied both methods to the Palmers Pinguins dataset and compared results in this `Notebook <https://forest-guided-clustering.readthedocs.io/en/latest/_tutorials/comparing_FGC_to_feature_importance.html>`_ .

**Citation:** If Forest-Guided Clustering is useful for your research, consider citing the package via `DOI: 10.5281/zenodo.7085465 <https://zenodo.org/badge/latestdoi/397931780>`_.


.. toctree::
   :maxdepth: 1
   :caption: GETTING STARTED

   _getting_started/installation.rst
   _getting_started/basic_usage.rst


.. toctree::
   :maxdepth: 1
   :caption: THEORETICAL BACKGROUND

   _theoretical_background/introduction.rst
   _theoretical_background/general_algorithm.rst
   _theoretical_background/feature_importance.rst


.. toctree::
   :maxdepth: 1
   :caption: TUTORIALS

   _tutorials/introduction_to_FGC
   _tutorials/comparing_FGC_to_feature_importance
   _tutorials/inference_with_FGC
   _tutorials/impact_of_model_complexity_on_FGC
   _tutorials/big_data_with_FGC


.. toctree::
   :maxdepth: 2
   :caption: API Documentation:

   fgclustering.rst


License
--------

The fgclustering package is MIT licensed.

Contributing
-------------

Contributions are more than welcome! Everything from code to notebooks to examples and documentation are all equally valuable so please don't feel you can't contribute. 
To contribute please fork the project make your changes and submit a pull request. We will do our best to work through any issues with you and get your code merged into the main branch.
