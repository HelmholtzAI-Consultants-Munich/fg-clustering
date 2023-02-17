Installation
==================

Requirements
--------------

This package was tested for ``Python 3.7 - 3.11`` on ubuntu, macos and windows. It depends on the ``kmedoids`` python package. If you are using windows or macos, you may need to first install Rust/Cargo with:

::

    conda install -c conda-forge rust


If this does not work, please try to install Cargo from source:

::

    git clone https://github.com/rust-lang/cargo
    cd cargo
    cargo build --release


For further information on the kmedoids package, please visit `this page <https://pypi.org/project/kmedoids/>`__.

All other required packages are automatically installed if installation is done via ``pip``.


Install Options
-----------------

The installation of the package is done via pip. Note: if you are using conda, first install pip with: ``conda install pip``.

PyPI install:

::

    pip install fgclustering


Installation from source:

::

    git clone https://github.com/HelmholtzAI-Consultants-Munich/fg-clustering.git


- Installation as python package (run inside directory):
    
    ::

		pip install .   


- Development Installation as python package (run inside directory):
    
    ::

		pip install -e . [dev]