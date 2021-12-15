from distutils.core import setup

setup(
    name='ForestGuidedClustering',
    version='0.1.dev0',
    packages=setuptools.find_packages(),
    install_requires=['pandas','numpy','matplotlib','seaborn','sklearn','scikit-learn-extra','scipy','tqdm', 'statsmodels'],
    docs_extras = [
        'Sphinx = 4.2.0',  # Force RTD to use >= 3.0.0
    ]
    long_description=open('README.md').read(),
)
