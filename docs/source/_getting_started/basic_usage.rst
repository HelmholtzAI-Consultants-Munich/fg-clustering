Basic Usage
================================================

To get explainability of your Random Forest model via Forest-Guided Clustering, you simply need to run the following commands:

..  code-block:: python

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

- ``model=rf`` is a trained Random Forest Classifier or Regressor object,
- ``data=data`` is a dataset containing the same features as required by the Random Forest model, and
- ``target_column='target'`` is the name of the target column (i.e. *target*) in the provided dataset. 

For detailed instructions, please have a look at :doc:`../_tutorials/introduction_to_FGC_use_cases`.

**Usage on big datasets**

If you are working with the dataset containing large number of samples, you can use one of the following strategies:

- Use the cores you have at your disposal to parallelize the optimization of the cluster number. You can do so by setting the parameter ``n_jobs`` to a value > 1 in the ``run()`` function.
- Use the faster implementation of the pam method that K-Medoids algorithm uses to find the clusters by setting the parameter  ``method_clustering`` to *fasterpam* in the ``run()`` function.
- Use subsampling technique

For detailed instructions, please have a look at :doc:`../_tutorials/special_case_big_data_with_FGC`.
