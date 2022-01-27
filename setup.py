from distutils.core import setup
import setuptools

setup(
    name='fgclustering',
    version='0.1.2',
    url='https://github.com/HelmholtzAI-Consultants-Munich/forest_guided_clustering',
    author='Lisa Barros de Andrade e Sousa, Dominik Thalmeier',
    author_email='lisa.barros.andrade.sousa@gmail.com, dominikthalmeier@googlemail.com',
    packages=setuptools.find_packages(),
    install_requires=['pandas','numpy','matplotlib','seaborn','sklearn','scikit-learn-extra','scipy','tqdm', 'statsmodels','Sphinx==4.2.0'],
    long_description=open('README.rst').read(),
)
