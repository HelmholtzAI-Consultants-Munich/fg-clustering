from distutils.core import setup

setup(
    name='ForestGuidedClustering',
    version='0.1.dev0',
    packages=setuptools.find_packages(),
    install_requires=['pandas','numpy','matplotlib','seaborn','sklearn','scikit-learn-extra','scipy','tqdm'],
    long_description=open('README.md').read(),
)