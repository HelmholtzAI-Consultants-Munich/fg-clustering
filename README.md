<div align="center">

<img src="https://raw.githubusercontent.com/HelmholtzAI-Consultants-Munich/fg-clustering/main/docs/source/_figures/FGC_Logo.png" width="200">
	

# Forest-Guided Clustering - Explainability for Random Forest Models


![stability-stable](https://img.shields.io/badge/stability-stable-green.svg)
[![stars](https://img.shields.io/github/stars/HelmholtzAI-Consultants-Munich/forest_guided_clustering?logo=GitHub&color=yellow)](https://github.com/HelmholtzAI-Consultants-Munich/forest_guided_clustering/stargazers)
[![PyPI](https://img.shields.io/pypi/v/fgclustering.svg)](https://pypi.org/project/fgclustering)
[![testing](https://github.com/HelmholtzAI-Consultants-Munich/fg-clustering/actions/workflows/testing.yml/badge.svg)](https://github.com/HelmholtzAI-Consultants-Munich/fg-clustering/actions/workflows/testing.yml)
[![docs](https://readthedocs.org/projects/forest-guided-clustering/badge/?version=latest)](https://forest-guided-clustering.readthedocs.io)
[![cite](https://zenodo.org/badge/397931780.svg)](https://zenodo.org/badge/latestdoi/397931780)
	
[Docs] | [Tutorials]

[Docs]: https://forest-guided-clustering.readthedocs.io/en/latest/
[Tutorials]: https://github.com/HelmholtzAI-Consultants-Munich/fg-clustering/tree/main/tutorials

</div>

This python package is about explainability of Random Forest models. Standard explainability methods (e.g. feature importance) assume independence of model features and hence, are not suited in the presence of correlated features. The Forest-Guided Clustering algorithm does not assume independence of model features, because it computes the feature importance based on subgroups of instances that follow similar decision rules within the Random Forest model. Hence, this method is well suited for cases with high correlation among model features. To showcase the advantages of FGC over Feature Importance we applied both methods to the Palmers Pinguins dataset and compared results in this [notebook](https://github.com/HelmholtzAI-Consultants-Munich/fg-clustering/blob/main/tutorials/comparing_FGC_to_feature_importance.ipynb).

For detailed documentation and usage examples, please visit the [Read the Docs documentation](https://forest-guided-clustering.readthedocs.io/).


## Installation

### Requirements

This packages was tested for ```Python 3.7 - 3.11``` on ubuntu, macos and windows. It depends on the ```kmedoids``` python package. If you are using windows or macos, you may need to first install Rust/Cargo with:

```
conda install -c conda-forge rust
```

If this does not work, please try to install Cargo from source:

```
git clone https://github.com/rust-lang/cargo
cd cargo
cargo build --release
```

For further information on the kmedoids package, please visit [this page](https://pypi.org/project/kmedoids/).

All other required packages are automatically installed if installation is done via ```pip```.


### Install Options

The installation of the package is done via pip. Note: if you are using conda, first install pip with: ```conda install pip```.

PyPI install:

```
pip install fgclustering
```


Installation from source:

```
git clone https://github.com/HelmholtzAI-Consultants-Munich/fg-clustering.git
```

- Installation as python package (run inside directory):

		pip install .   


- Development Installation as python package (run inside directory):

		pip install -e . [dev]


## Basic Usage

To get explainability of your Random Forest model via Forest-Guided Clustering, you simply need to run the following commands:

```python
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
```

where 

- ```model=rf``` is a Random Forest Classifier or Regressor object,
- ```data=data``` is a dataset containing the same features as required by the Random Forest model, and
- ```target_column='target'``` is the name of the target column (i.e. *target*) in the provided dataset. 

For a detailed tutorial see the IPython Notebook [Introduction to FGC](https://github.com/HelmholtzAI-Consultants-Munich/fg-clustering/blob/main/tutorials/introduction_to_FGC.ipynb).

**Usage on big datasets**

If you are working with the dataset containing large number of samples, you can use one of the following strategies:

- Use the cores you have at your disposal to parallelize the optimization of the cluster number. You can do so by setting the parameter ```n_jobs``` to a value > 1 in the ```run()``` function.
- Use the faster implementation of the pam method that K-Medoids algorithm uses to find the clusters by setting the parameter  ```method_clustering``` to *fasterpam* in the ```run()``` function.
- Use subsampling technique

For a detailed tutorial on the usage on big datasets, please see the IPython Notebook [FGC for Big Datasets](https://github.com/HelmholtzAI-Consultants-Munich/fg-clustering/blob/main/tutorials/big_data_with_FGC.ipynb).

## License

The fgclustering package is MIT licensed.


## Contributing
 
Contributions are more than welcome! Everything from code to notebooks to examples and documentation are all equally valuable so please don't feel you can't contribute. To contribute please fork the project make your changes and submit a pull request. We will do our best to work through any issues with you and get your code merged into the main branch.

## How to cite

If Forest-Guided Clustering is useful for your research, consider citing the package:

```
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
```