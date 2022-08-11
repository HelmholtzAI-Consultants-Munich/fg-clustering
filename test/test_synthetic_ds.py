import joblib
import pandas as pd
import sys
import os
from fgclustering.forest_guided_clustering import *
import matplotlib.pyplot as plt


def plot_results(x,y, c_original, c_clustering, plot_filepath, title):

    # Save the original plot and the plot with fg clustering output:
    plt.figure(figsize=(8, 8))
    plt.suptitle(title)
    plt.subplots_adjust(bottom=0.05, top=0.9, left=0.05, right=0.95)
    plt.subplot(121)
    plt.title("Original:", fontsize="small")
    plt.scatter(x, y, marker="o", c=c_original, s=25, edgecolor="k")
    plt.subplot(122)
    plt.title("FGC:", fontsize="small")
    plt.scatter(x, y, marker="o", c=c_clustering, s=25, edgecolor="k")
    
    plt.savefig(plot_filepath)


def test_make_classif_fgc():

    # -- setup 1 -- #
    data_1 = pd.read_csv('./data/data_make_classif_1.csv')
    model_1 = joblib.load('./data/random_forest_make_classif_1.joblib')
    
    expected_output = 2

    # initialize and run fgclustering object
    fgc = FgClustering(model=model_1, data=data_1, target_column='target')
    fgc.run(max_K = 8, bootstraps_JI = 30, max_iter_clustering = 100, discart_value_JI = 0.6)

    data_1['cluster_labels'] = fgc.cluster_labels

    # Save the original plot and the plot with fg clustering output:
    plot_results(data_1['X1'], data_1['X2'], data_1['target'], data_1['cluster_labels'],  
    './test/make_classification_1.jpg', "One informative feature, one cluster per class.")

    print(f'True number of clusters is {2}, and the fgc-optimized number of clusters is {fgc.k}')

    print(pd.crosstab(data_1['target'], data_1['cluster_labels']))

    #assert result == expected_output

    # -- setup 2 -- #
    print(" ----------- SETUP 2 -----------")
    data_2 = pd.read_csv('./data/data_make_classif_2.csv')
    model_2 = joblib.load('./data/random_forest_make_classif_2.joblib')

    fgc = FgClustering(model=model_2, data=data_2, target_column='target')
    fgc.run(max_K = 8, bootstraps_JI = 30, max_iter_clustering = 100, discart_value_JI = 0.6)

    data_2['cluster_labels'] = fgc.cluster_labels
    print(f'True number of clusters is {2}, and the fgc-optimized number of clusters is {fgc.k}')

    print(pd.crosstab(data_2['target'], data_2['cluster_labels']))

    #assert fgc.k == 2


test_make_classif_fgc()