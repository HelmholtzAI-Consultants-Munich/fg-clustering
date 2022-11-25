from distutils.core import setup
import setuptools

setup(
    name='fgclustering',
    version='1.0.0',
    url='https://github.com/HelmholtzAI-Consultants-Munich/forest_guided_clustering',
    author='Lisa Barros de Andrade e Sousa, Dominik Thalmeier, Helena Pelin, Marie Piraud',
    author_email='lisa.barros.andrade.sousa@gmail.com, dominikthalmeier@googlemail.com, helena.pelin@helmholtz-muenchen.de',
    packages=setuptools.find_packages(),
    install_requires=['pandas','numpy','matplotlib','seaborn','sklearn','kmedoids','scipy','tqdm', 'statsmodels>=0.13.5','Sphinx>=4.2.0', 'numexpr>=2.8.4'],
    long_description=open('README.rst').read(),
)
