Feature Importance
===============

For each cluster we want to figure out a in which features this cluster is significanlty different from the total data.

For continuous features we check if the variance of a particular feature is signifiantly smaller for a cluster compared to the total data.
For categorical features we look if the impurity score of a feature (see :doc:`general_algorithm` for the definition of the impurity score) is significantly lower for the cluster
compared to the total data.
We compute the p-value for the null hypothesis that the test statistic for this cluster is not smaller than the test statistics computed on the total dataset for this particular feature.
A small p-value shows that the cluster is significantly different from the rest of the data regarding that feature. 
We therefore define the feature importance as 1-p, where a high feature importance shows that this feature allows to discriminate this cluster from a randomly selected subset of the total dataset.

Computation of the feature importance using bootstrapping
--------------------------------------------
We compute the p-value using a bootstrapping approach.

For a cluster :math:`A` with M datapoints we computethe test statistics T (variance in case of continuous variables or impurity in case of categorical variables) on feature
X:
:math:`T(X_A)`.

To test if the statistics :math:`T(X_A)` allows to disriminate this cluster against the total dataset we boostrap N random subsets :math:`B_n` of the same size M
as the cluster A from the total dataset.

The proportion of boostrapped subset for this the teststatistics :math:`T(X_{B_n})` is smaller than the test statistics :math:`T(X_A)` is an estimate for the
p-value.

Thus we get 

:math:`\text{feature_importance} = 1-\frac{Inf(T(X_{B_n})<T(X_A))}{N}`,

with :math:`Ind()` being the indicator function that is equal one when its argument is true and zero otherwise.
