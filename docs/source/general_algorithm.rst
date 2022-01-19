General Algorithm
===============

Compute distance matrix between data points
--------------------------------------------
The first step in the algorithm is to define a distance matrix that measures the distance between the data points.

A random forest consists of :math:`N` decision trees. Each datapoints for which a classification or prediction is made traverses each of these decision trees
and ends up in a leave of the tree that specifies the prediction of the tree. 
We say that two datapoints are similar, when they end up in the same leave of a tree.
This similarity can be quantitively measured by counting the number of trees :math:`m_{i,j}` trees two data points :math:`i,j` end up.
We define the proximity matrix as :math:`M^{proximity}_{ij} = \frac{m_{i,j}}{N}`.

Based on this we can define a distance matrix as 
:math:`M^{distance}_{ij} = 1-M^{proximity}_{ij}`


Forest guided clustering
------------------------
Having a distance matrix :math:`M^{distance}_{ij}` we can use  `k-medioids clustering <https://en.wikipedia.org/wiki/K-medoids>`_:
to find subgroups of the data for which the data points follow similar decision paths in the random forest.
In contrast to k-means, k-medioids does not require an embedding of the data points in a metric space
but can be applied if only a matrix of the distances between the data points is available.

**Optimization of number of clusters**

Similar to k-means, we still need to specify the number of clusters :math:`k` into which we want to divide our dataset.
We want the clusters that we find to be both stable and predictive for the target.
Therefore we compute a clustering for a range of :math:`k` (specified by the user) and
1. only allow clusterings that follow stability criterion
2. minimize a cost that judges how well the clusters can be used to differentiate between the target values 

*Compute cluster stability*
We measure the stability of the clustering using the Jaccard index.
The Jaccard Index measures how similar a clustering stays when data-set is bootstrapped n-times.
𝐽𝑎𝑐𝑐𝑎𝑟𝑑𝐼𝑛𝑑𝑒𝑥(𝐴)=(∑_(𝑏=1)^𝑛▒|𝐴∩𝐵_𝑏 |/|𝐴∪𝐵_𝑏 | )/𝑛


We measure the Jaccard Index of each cluster: Average overlap of cluster A with its bootstrapped siblings   
Only when all clusters are stable (Jaccard index > 0.6 [1]) a clustering with k clusters is considered valid



an discard any clustering with an Jaccard index :math:`>0.6` (see 

We compute a k-medioids clustering

*Compute predictive power of clusters*
For a classification problem we want to choose the 


*cluster stability*

*error*

- classification
- regression


Visualization
------------------------

**heat map**

**box plots**


**feature importance**
