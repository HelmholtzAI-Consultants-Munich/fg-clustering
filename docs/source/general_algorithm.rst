General Algorithm
===============

Compute distance matrix between data points
--------------------------------------------
The first step in the algorithm is to define a distance matrix that measures the distance between the data points.

The proximity measure between two instances i and j represents the
frequency with which those instances occur in the same terminal nodes of a
tree in the Random Forest(RF), intuitively defining how close those instances are in the RF
model.
We define the proximity matrix as :math:`M^{proximity}_{ij} = \frac{m_{i,j}}{N}`, where :math:`m_{i,j}` is the number of trees where the data-points i,j end up in the same terminal node and N is the total number of trees in the RF.
According to *Breiman et al., 2003* the values :math:`1-M^{proximity}_{ij}` are square distances in a euclidean space and can therefore be used as distance measures:
:math:`M^{distance}_{ij} = 1-M^{proximity}_{ij}`


Forest guided clustering
------------------------
Having a distance matrix :math:`M^{distance}_{ij}` we can use  `k-medioids clustering <https://en.wikipedia.org/wiki/K-medoids>`_:
to find subgroups of the data for which the data points follow similar decision paths in the random forest.
In contrast to k-means, k-medioids does not require an embedding of the data points in a metric space
but can be applied if only a matrix of the distances between the data points is available.

**Optimization of number of clusters**

Similar to k-means clustering, k-medoids clustering requires setting the number of clusters :math:`k` into which we want to divide our dataset.
We want the clusters that we find to be both stable and predictive for the target.
We developed a scoring system to choose
the optimal number of clusters :math:`k`, which minimizes the model bias while
restricting the model complexity. The model bias measures how well the
trained model (with a certain value of :math:`k`) approximates the expected model,
while the variance is related to the model complexity, since complex models
usually have a high variance and poor generalization capability.

*Model bias*

regression

classification

*Model variance*
We restrict the model variance by discarding too complex models. We define
the complexity of the clustering for each value of :math:`k` by its stability. The
stability of each cluster :math:`i` in the clustering is measured by the average Jaccard
Similarity between the original cluster :math:`A` and :math:`n` bootstrapped clusters :math:`B_b`:

:math:`JS_i(A) = \frac{\sum_{b=1}^n\frac{A}{B}}{n}'

Jaccard similarity values > 0.6 are usually indicative of stable patterns in the
data (*Hennig, 2008*). Only stable clusterings, i.e. clustering with low variance,
are considered as clustering candidates. Hence, the optimal number of
clusters :math:`k` is the one yielding the minimum model bias, while having a stable
clustering.



Visualization
------------------------

**heat map**

**box plots**


**feature importance**
