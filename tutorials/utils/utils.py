import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_permutation_feature_importance(result, data, title, figsize=(5, 4)):
    # Sort the features by importance mean
    perm_sorted_idx = result.importances_mean.argsort()[::-1]

    # Prepare the data for Seaborn's boxplot (convert to long format)
    feature_importances = result.importances[perm_sorted_idx].T
    df = pd.DataFrame(feature_importances, columns=data.columns[perm_sorted_idx])
    df_long = df.melt(var_name="Feature", value_name="Importance")

    # Create the figure and plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df_long, x="Importance", y="Feature", ax=ax, flierprops=dict(marker=".", alpha=0.5, markersize=2)
    )

    # Set title and layout
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_impurity_feature_importance(importance, names, title, figsize=(5, 4)):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=figsize)
    # Plot Searborn bar chart
    sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"], color="#3470a3")
    # Add chart labels
    plt.title(title)
    plt.xlabel("feature importance")
    plt.ylabel("feature names")


def plot_correlation_matrix(data, figsize=(5, 5)):
    f, ax = plt.subplots(figsize=figsize)
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    np.fill_diagonal(mask, False)
    sns.heatmap(
        round(corr, 2),
        mask=mask,
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        square=True,
        ax=ax,
        annot=True,
    )
