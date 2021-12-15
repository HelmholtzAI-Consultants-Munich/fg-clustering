from distutils.core import setup

docs_extras = [
        'Sphinx = 4.2.0',  # Force RTD to use >= 3.0.0
    ]

setup(
    name='ForestGuidedClustering',
    version='0.1.dev0',
    packages=setuptools.find_packages(),
    install_requires=['pandas','numpy','matplotlib','seaborn','sklearn','scikit-learn-extra','scipy','tqdm', 'statsmodels'],
    extras_require={'docs': docs_extras},
    long_description=open('README.md').read(),
)
