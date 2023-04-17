.. Forest Guided Clustering documentation master file, created by
   sphinx-quickstart on Thu Dec  9 16:39:21 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*Forest-Guided Clustering* - Shedding light into the Random Forest Black Box 
==============================================================================

Forest-Guided Clustering (FGC) is an explainability method for Random Forest models. 
Standard explainability methods (e.g. feature importance) assume independence of model features and hence, 
are not suited in the presence of correlated features. The Forest-Guided Clustering algorithm does not assume independence of model features, 
because it computes the feature importance based on subgroups of instances that follow similar decision rules within the Random Forest model. 
Hence, this method is well suited for cases with high correlation among model features.  

For a short introduction to Forest-Guided Clustering, click below:

.. vimeo:: 746443233?h=07ddf2290b

For a detailed comparison of FGC and Permutation Feature Importance, have a look at this Notebook :doc:`../_tutorials/introduction_to_FGC_comparing_FGC_to_FI`.



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

   _tutorials/introduction_to_FGC_use_cases
   _tutorials/introduction_to_FGC_comparing_FGC_to_FI
   _tutorials/special_case_inference_with_FGC
   _tutorials/special_case_impact_of_model_complexity_on_FGC
   _tutorials/special_case_big_data_with_FGC


.. toctree::
   :maxdepth: 2
   :caption: API DOCUMENTATION:

   fgclustering.rst


Contributing
-------------

Contributions are more than welcome! Everything from code to notebooks to examples and documentation are all equally valuable so please don't feel you can't contribute. 
To contribute please fork the project make your changes and submit a pull request. We will do our best to work through any issues with you and get your code merged into the main branch.


How to cite
-------------

If Forest-Guided Clustering is useful for your research, consider citing the package:

::

   @software{lisa_sousa_2022_7823042,
       author       = {Lisa Barros de Andrade e Sousa,
                        Helena Pelin,
                        Dominik Thalmeier,
                        Marie Piraud},
       title        = {{Forest-Guided Clustering - Explainability for Random Forest Models}},
       month        = april,
       year         = 2022,
       publisher    = {Zenodo},
       version      = {v1.0.3},
       doi          = {10.5281/zenodo.7823042},
       url          = {https://doi.org/10.5281/zenodo.7823042}
   }
 
License
--------

``fgclustering`` is released under the MIT license. See `LICENSE <https://github.com/HelmholtzAI-Consultants-Munich/fg-clustering/blob/main/LICENSE>`_ for additional details about it.

