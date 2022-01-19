General Algorithm
===============

We define a similarity between two data points 

The first step in the algorithm is to define a proximity matrix that measures the distance between the data points.

A random forest consists of $N$ decision trees. Each datapoints for which a classification or prediction is made traverses each of these decision trees
and ends up in a leave of the tree that specifies the prediction of the tree. 
We say that two datapoints are similar, when they end up in the same leave of a tree.
This similarity can be quantitively measured by counting the number of trees $m_{i,j}$ trees two data points $i,j$ end up.
We define the proximity matrix as

:math:`a^2 + b^2 = c^2`

:math:`\\frac{1}{2}`

:raw-math:`$$ \frac{s}{\sqrt{N}} $$`

.. math::
  M^\text{proximity}_{ij} = \frac{m_{i,j}{N}


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
