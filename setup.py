from distutils.core import setup

setup(
    name='ForestGuidedClustering',
    version='0.1.dev0',
    packages=setuptools.find_packages(),
    install_requires=['pandas','numpy','matplotlib','seaborn','sklearn','scikit-learn-extra','scipy','tqdm', 'statsmodels'],
    docs_extras = [
        'Sphinx >= 3.0.0',  # Force RTD to use >= 3.0.0
        'docutils',
        'pylons-sphinx-themes >= 1.0.8',  # Ethical Ads
        'pylons_sphinx_latesturl',
        'repoze.sphinx.autointerface',
        'sphinx-copybutton',
        'sphinxcontrib-autoprogram',
    ],
    long_description=open('README.md').read(),
)
