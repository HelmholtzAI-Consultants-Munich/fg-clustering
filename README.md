<div align="center">

<img src="https://raw.githubusercontent.com/HelmholtzAI-Consultants-Munich/fg-clustering/main/docs/source/_figures/FGC_Logo.png" width="200">
	

# *Forest-Guided Clustering* - Shedding light into the Random Forest Black Box 

[![Docs](https://img.shields.io/badge/docs-latest-blue?style=flat&logo=readthedocs)](https://forest-guided-clustering.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/fgclustering.svg)](https://pypi.org/project/fgclustering)
[![PyPI Downloads](https://static.pepy.tech/badge/fgclustering)](https://pepy.tech/projects/fgclustering)
[![stars](https://img.shields.io/github/stars/HelmholtzAI-Consultants-Munich/forest_guided_clustering?logo=GitHub&color=yellow)](https://github.com/HelmholtzAI-Consultants-Munich/forest_guided_clustering/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2507.19455-b31b1b.svg)](https://doi.org/10.48550/arXiv.2507.19455)
[![test](https://github.com/HelmholtzAI-Consultants-Munich/fg-clustering/actions/workflows/test.yml/badge.svg)](https://github.com/HelmholtzAI-Consultants-Munich/fg-clustering/actions/workflows/test.yml)
	
</div>

## ✨ About this Package

<!-- LINK INTRODUCTION START -->

**Why Use Forest-Guided Clustering?**

Forest-Guided Clustering (FGC) is an explainability method for Random Forest models that addresses one of the key limitations of many standard XAI techniques: the inability to effectively handle correlated features and complex decision patterns. Traditional methods like permutation importance, SHAP, and LIME often assume feature independence and focus on individual feature contributions, which can lead to misleading or incomplete explanations. As machine learning models are increasingly deployed in sensitive domains like healthcare, finance, and HR, understanding why a model makes a decision is as important as the decision itself. This is not only a matter of trust and fairness, but also a legal requirement in many jurisdictions, such as the European Union's GDPR which mandates a “right to explanation” for automated decisions.

FGC offers a different approach: instead of approximating the model with simpler surrogates, it uses the internal structure of the Random Forest itself. By analyzing the tree traversal patterns of individual samples, FGC clusters data points that follow similar decision paths. This reveals how the forest segments the input space, enabling a human-interpretable view of the model's internal logic. FGC is particularly useful when features are highly correlated, as it does not rely on assumptions of feature independence. It bridges the gap between model accuracy and model transparency, offering a powerful tool for global, model-specific interpretation of Random Forests.

**📢 New! Forest-Guided Clustering is now on arXiv**

Please see our paper [Forest-Guided Clustering - Shedding Light into the Random Forest Black Box](https://doi.org/10.48550/arXiv.2507.19455) for a detailed description of the method, its theoretical foundations, and practical applications. Check it out to learn more about how FGC reveals structure in your Random Forest models!

**Prefer a visual walkthrough?**
Watch our short introduction video by clicking below:

<div align="center">

[![Video](http://i.vimeocdn.com/video/1501376117-3e402fde211d1a52080fb16b317efc3786a34d0be852a81cfe3a03aa89adc475-d_295x166)](https://vimeo.com/746443233/07ddf2290b)

</div>

**Curious how Forest-Guided Clustering compares to standard methods?**
See our notebook: [Introduction to FGC: Comparison of Forest-Guided Clustering and Feature Importance](https://github.com/HelmholtzAI-Consultants-Munich/fg-clustering/blob/main/tutorials/introduction_to_FGC_comparing_FGC_to_FI.ipynb).

<!-- LINK INTRODUCTION END -->

**Want to dive deeper?**
Visit our [full documentation](https://forest-guided-clustering.readthedocs.io/) for:

- Getting Started – Installation and quick start
- Tutorials – Use cases for classification, regression, and large datasets
- API Reference – Detailed descriptions of functions and classes

## 🛠️ Installation

<!-- LINK INSTALLATION START -->

**Requirements**

This package was tested for `Python 3.8 - 3.13` on ubuntu, macos and windows. It depends on the `kmedoids` python package. If you are using windows or macos, you may need to first install Rust/Cargo with:

    conda install -c conda-forge rust


If this does not work, please try to install Cargo from source:

    git clone https://github.com/rust-lang/cargo
    cd cargo
    cargo build --release


For further information on the kmedoids package, please visit [this page](https://pypi.org/project/kmedoids/).

All other required packages are automatically installed if installation is done via `pip`.


**Install Options**

The installation of the package is done via pip. Note: if you are using conda, first install pip with: `conda install pip`.

PyPI install:

    pip install fgclustering


Installation from source:

    git clone https://github.com/HelmholtzAI-Consultants-Munich/fg-clustering.git


- Installation as python package (run inside directory):

		pip install .   


- Development Installation as python package (run inside directory):

		pip install -e .

<!-- LINK INSTALLATION END -->

## 💻 How to Use Forest-Guided Clustering

<!-- LINK BASIC USAGE START -->

**Basic Usage**

To apply Forest-Guided Clustering (FGC) for explaining a Random Forest model, you can follow the simple workflow consisting of three main steps: computing the forest-guided clusters, evaluating feature importance, and visualizing the results.

```python
# compute the forest-guided clusters
fgc = forest_guided_clustering(
    estimator=model, 
    X=X, 
    y=y, 
    clustering_distance_metric=DistanceRandomForestProximity(), 
    clustering_strategy=ClusteringKMedoids(),
)

# evaluate feature importance
feature_importance = forest_guided_feature_importance(
    X=X, 
    y=y, 
    cluster_labels=fgc.cluster_labels,
    model_type=fgc.model_type,
)

# visualize the results
plot_forest_guided_feature_importance(
    feature_importance_local=feature_importance.feature_importance_local,
    feature_importance_global=feature_importance.feature_importance_global
)

plot_forest_guided_decision_paths(
    data_clustering=feature_importance.data_clustering,
    model_type=fgc.model_type,
)
```

where
- `estimator` is the trained Random Forest model
- `X` is the feature matrix
- `y` is the target variable
- `clustering_distance_metric` defines how similarity between samples is measured based on the Random Forest structure
- `clustering_strategy` determines how the proximity-based clustering is performed 

For a detailed walkthrough, refer to the [Introduction to FGC: Simple Use Cases](https://github.com/HelmholtzAI-Consultants-Munich/fg-clustering/blob/main/tutorials/introduction_to_FGC_use_cases.ipynb) notebook.


**Using FGC on Large Datasets**

When working with datasets containing a large number of samples, Forest-Guided Clustering (FGC) provides several strategies to ensure efficient performance and scalability:

* *Parallelize Cluster Optimization*: Leverage multiple CPU cores by setting the `n_jobs` parameter to a value greater than 1 in the `forest_guided_clustering()` function. This will parallelize the bootstrapping process for evaluating cluster stability.

* *Use a Faster Clustering Algorithm*: Improve the efficiency of the K-Medoids clustering step by using the optimized `"fasterpam"` algorithm. Set the `method` parameter of your clustering strategy (e.g., `ClusteringKMedoids(method="fasterpam")`) to activate this faster implementation.

* *Enable Subsampling with CLARA*: For extremely large datasets, consider using the CLARA (Clustering Large Applications) variant by choosing `ClusteringClara()` as your clustering strategy. CLARA performs clustering on smaller random subsamples, making it suitable for high-volume data.

For a detailed example, please refer to the notebook [Special Case: FGC for Big Datasets](https://github.com/HelmholtzAI-Consultants-Munich/fg-clustering/blob/main/tutorials/special_case_big_data_with_FGC.ipynb).

<!-- LINK BASIC USAGE END -->

## 🤝 Contributing

<!-- LINK CONTRIBUTION START -->
 
We welcome contributions of all kinds—whether it’s improvements to the code, documentation, tutorials, or examples. Your input helps make Forest-Guided Clustering more robust and useful for the community.

To contribute:

1. Fork the repository.
2. Make your changes in a feature branch.
3. Submit a pull request to the main branch.

We’ll review your submission and work with you to get it merged.

If you have any questions or ideas you'd like to discuss before contributing, feel free to reach out to [Lisa Barros de Andrade e Sousa](mailto:lisa.barros@helmholtz-munich.de).

<!-- LINK CONTRIBUTION END -->

## 📝 How to cite

<!-- LINK CITE START -->

If you find Forest-Guided Clustering useful in your research or applications, please consider citing it:

```
@article{barros2025forest,
    title	= {Forest-Guided Clustering -- Shedding Light into the Random Forest Black Box},
    author	= {Lisa Barros de Andrade e Sousa,
		   Gregor Miller,
		   Ronan Le Gleut,
		   Dominik Thalmeier,
		   Helena Pelin,
		   Marie Piraud},
    journal	= {ArXiv},
    year	= {2025},
    url         = {https://doi.org/10.48550/arXiv.2507.19455}
}

```

<!-- LINK CITE END -->

## 🛡️ License

<!-- LINK LICENSE START -->

The `fgclustering` package is released under the MIT License. You are free to use, modify, and distribute it under the terms outlined in the [LICENSE](https://github.com/HelmholtzAI-Consultants-Munich/fg-clustering/blob/main/LICENSE) file.

<!-- LINK LICENSE END -->
