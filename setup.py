from distutils.core import setup

setup(
    name='ForestGuidedClustering',
    version='0.1.0',
    packages=setuptools.find_packages(),
    install_requires=['pandas','numpy','matplotlib','seaborn','sklearn','scikit-learn-extra','scipy','tqdm', 'statsmodels','Sphinx==4.2.0'],
    long_description=open('README.rst').read(),
)
