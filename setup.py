from distutils.core import setup

setup(
    name='ForestGuidedClustering',
    version='0.1.dev0',
    packages=setuptools.find_packages(),
    install_requires=['pandas','numpy','matplotlib','sklearn','scikit-learn-extra','tqdm'],
    long_description=open('README.md').read(),
)