Feature Importance
===================

Global Feature Importance
--------------------------
We measure the global feature importance by the significance of the difference between cluster-wise feature distributions. Features, which have significantly different distributions across clusters, have a high feature importance, while features, which have a similar feature distribution across clusters have a low feature importance. Those features are considered to be less important in the decision making process of the Random Forest model, because no clear rules can be derived from those features.

For continuous variables, we apply an Anova test to test if the cluster-wise feature mean is varying significantly between the clusters. For categorical variables, we use a Chi-Square test to test if there is a relation of the feature across clusters. For feature visualization we only consider features that are below a significance threshold that is adjustable by the user (default value is :math:`p\leq 0.01`).

Since a small p-value indicates a relevant feature, we define the global feature importance as :math:`\text{global_feature_importance} = 1-p-value`


Local Feature Importance
--------------------------

Forest-Guided Clustering allows to split the data into clusters that follow similar decision paths in the Random Forest model. In order to understand the relevance of each feature for a specific cluster, we compute a local ( = cluster-wise) feature importance metric.

We define that the local feature importance for a particular feature in cluster is high, if a statistical test indicates with strong significance, that the feature values attributed to this cluster are not randomly drawn from the full feature values. For *continuous features*, we test if the variance of a particular feature is signifiantly smaller for a cluster compared to the full dataset. For *categorical features*, we test if the impurity score of a feature (see :doc:`general_algorithm` for the definition of the impurity score) is significantly lower for the cluster compared to the full dataset.

We compute the p-value for the null hypothesis (the test statistic for this cluster is not smaller than the test statistics for the total dataset) of this particular feature. The p-value is computed using bootstrapping. For a cluster :math:`A` with :math:`n_A` datapoints we compute the test statistics T (variance in case of continuous variables or impurity in case of categorical variables) on feature X: :math:`T(X_A)`. To test if the statistics :math:`T(X_A)` allows to discriminate this cluster against the full dataset we boostrap b random subsets :math:`B_b` of size :math:`n_A` (same size as the cluster :math:`A`) from the full dataset. The p-value is then defined as the proportion of the boostrapped subsets for which :math:`T(X_{B_b})<T(X_A)`. Thus we get 

:math:`\text{local_feature_importance} = 1-\frac{\sum_{b=1}^n Ind\left(T(X_{B_n})<T(X_A)\right)}{n}`

with :math:`Ind()` being the indicator function that equals one when its argument is true and zero otherwise. A small p-value indicates that the cluster is significantly different from the rest of the data, regarding that feature. Analogous to the global feature importance we define the local feature importance as :math:`\text{local_feature_importance} = 1-p-value`, where a high feature importance shows that this feature allows to discriminate this cluster from a randomly selected subset of the total dataset.

