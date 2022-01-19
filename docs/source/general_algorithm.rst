General Algorithm
===============

We define a similarity between two data points 

The first step in the algorithm is to define a proximity matrix that measures the distance between the data points.

A random forest consists of $N$ decision trees. Each datapoints for which a classification or prediction is made traverses each of these decision trees
and ends up in a leave of the tree that specifies the prediction of the tree. 
We say that two datapoints are similar, when they end up in the same leave of a tree.
This similarity can be quantitively measured by counting in how many of the $N$ trees two data points end up.
We define the proximity matrix 
$$M^\text{proximity}_{ij} = $


Define similarity of two datapoints by counting in how many trees they end up in same terminal nodes
-> i.e. took same decision path


Compute proximity matrix
------------------------


Forest guided clustering
------------------------

**distance matrix, k-mediouds**

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
