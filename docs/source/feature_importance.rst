Feature Importance
===================

Global Feature Importance
--------------------------
We determine if a particular feature is actually varying between the clusters.
For continuous variable we apply an Anova test for continuous variables, to test if the means of the feature is significantly varying between the clusters.
For categorical variables we use a Chi-Square test.
For feature visualization we only consider features that are below a significance threshold that is adjustable by the user (default value is :math:`p\leq 0.01`).

Since a small p-value indicates a relevant feature, we define the global feature importance as
:math:`\text{local_feature_importance} = 1-p`


Local Feature Importance
--------------------------

Forest Guided clustering allows to split the data into clusters that are treated similarily by the random forest.
In order to understand which of the features are actually relevant for deciding in which cluster a data points ends up, we compute a local ( = cluster wise) feature importance metric.

We say that the local feature importance for a particular cluster is high, if a statistical test on this feature can determine with a strong significance, that the data points from the cluster are not randomly drawn from the full dataset.

For *continuous features* we test if the variance of a particular feature is signifiantly smaller for a cluster compared to the full dataset.
For *categorical features* we test if the impurity score of a feature (see :doc:`general_algorithm` for the definition of the impurity score) is significantly lower for the cluster compared to the full dataset.

We compute the p-value for the null hypothesis that the test statistic for this cluster is not smaller than the test statistics computed on the total dataset for this particular feature.
A small p-value shows that the cluster is significantly different from the rest of the data regarding that feature. 
Analogous to the global feature importance we define the local feature importance as 1-p, where a high feature importance shows that this feature allows to discriminate this cluster from a randomly selected subset of the total dataset.

Computation of the feature importance using bootstrapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The p-value is computed using bootstrapping.

For a cluster :math:`A` with M datapoints we compute the test statistics T (variance in case of continuous variables or impurity in case of categorical variables) on feature
X:
:math:`T(X_A)`.

To test if the statistics :math:`T(X_A)` allows to discriminate this cluster against the full dataset we boostrap n random subsets :math:`B_b` of size M (
same size as the cluster A) from the full dataset.

The p-value is then defined as the proportion of the boostrapped subsets for which :math:`T(X_{B_b})<T(X_A)`.
Thus we get 

:math:`\text{local_feature_importance} = 1-\frac{\sum_{b=1}^n Ind\left(T(X_{B_n})<T(X_A)\right)}{n}`,

with :math:`Ind()` being the indicator function that is equal one when its argument is true and zero otherwise.
