General Algorithm
===============

Compute distance matrix between data points
--------------------------------------------
The first step in the algorithm is to define a distance matrix that measures the distance between the data points.

A random forest consists of :math:`N` decision trees. Each datapoints for which a classification or prediction is made traverses each of these decision trees
and ends up in a leave of the tree that specifies the prediction of the tree. 
We say that two datapoints are similar, when they end up in the same leave of a tree.
This similarity can be quantitively measured by counting the number of trees :math:`m_{i,j}` trees two data points :math:`i,j` end up.
We define the proximity matrix as
:math:`M^{proximity}_{ij} = \frac{m_{i,j}}{N}`
Based on this we can define a distance matrix as 
:math:`M^{distance}_{ij} = 1-M^{proximity}_{ij}`


Forest guided clustering
------------------------
Based on this distance matrix :math:`M^{distance}_{ij}` we can use  `k-medioids clustering <https://en.wikipedia.org/wiki/K-medoids>`_:
to find subgroups of the data for which the data points follow similar decision paths in the random forest.
In contrast to k-means, k-medioids does not need an embedding of the datapoints in a metrix space
but can be applied when only a matrix of distances between the data points is available.

Similar to k-means however we need to specify the number of clusters :math:`k` in which we want to divide our dataset.
We want the clusters that we find to be both stable and predictive for the target.
Therefore we choose a :math:`k` that 
1. minimizes a cost that judges how well the clusters can be used to differentiate between the target values 
2. have a stability criterion for the clustering



**optimization of number of clusters**

*cluster stability threshold*

*error*

- classification
- regression


Visualization
------------------------

**heat map**

**box plots**


**feature importance**
