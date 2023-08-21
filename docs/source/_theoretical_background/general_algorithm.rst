General Algorithm
===================

Compute distance matrix between data points
--------------------------------------------
The first step in the algorithm is the computation of the distances between the data points. 
This distance matrix is based on a proximity measure between two instances i and j, that represents the frequency 
with which those instances occur in the same terminal nodes of a tree in the Random Forest (RF) model. 
Intuitively, this defines how close those instances are in the RF model. We define the proximity matrix as 

:math:`M^{proximity}_{ij} = \frac{m_{i,j}}{N}`

where :math:`m_{i,j}` is the number of trees where the data-points i,j end up in the same terminal node and N 
is the total number of trees in the RF model. According to *Breiman et al., 2003* the values :math:`1-M^{proximity}_{ij}` 
are square distances in a euclidean space and can therefore be used as distance measures: :math:`M^{distance}_{ij} = 1-M^{proximity}_{ij}`


Optimize number of clusters
-----------------------------
Having a distance matrix :math:`M^{distance}_{ij}` we can use  `k-medoids clustering <https://en.wikipedia.org/wiki/K-medoids>`_ 
to find subgroups for which the data points follow similar decision paths in the RF model. We use k-medoids, as, in contrast to k-means, 
it does not require an embedding of the data points in a metric space but can be applied directly to a distance matrix.

Similar to k-means clustering, k-medoids clustering requires setting the number of clusters :math:`k` into which we want 
to divide our dataset in advance. We want the resulting clusters to be both, stable and predictive of the target. 
We developed a scoring system to choose the optimal number of clusters :math:`k`, which minimizes the model bias while 
restricting the model complexity. The model bias measures how well the clustering (FGC with a certain value of :math:`k`) 
approximates the expected model, while the variance is related to the model complexity, since complex models 
usually have a high variance and poor generalization capability.

Model bias
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For **classification models**, we define the model bias by the impurity score of the clustering. 
Analogous to the regression case the model bias is computed for each value of :math:`k` separately. 
The impurity score is a balanced Gini coefficient of the classes within each cluster. 
The class sizes are balanced by rescaling the class size with the inverse size of the class in the overall dataset. 
Given a classification problem with :math:`G` classes, we define the impurity score as:
    
:math:`IS_k = \sum_i^k \left( 1- \sum_{g=1}^G b^2_{i,g} \right)` 

where the balanced per cluster frequency :math:`b_{i,g} = \frac{1}{\sum_{g=1}^G \frac{p_{i,g}}{q_g}} \frac{p_{i,g}}{q_g}` 
of class :math:`g` in cluster :math:`i` is the normalized frequency :math:`p_{i,g}` of class :math:`g` in cluster :math:`i`, 
weighted by the total frequency :math:`q_g` of class :math:`g` in the data set.

For **regression models**, the target value of each cluster is treated as the mean prediction values for each data point in the cluster. 
The model bias is then defined as the total squared error of this prediction compared to the ground truth. 
We compute the model bias separately for each value of :math:`k`. Then the clustering's are scored by the lowest total squared error:

:math:`TSE_k = \sum_i^k \sum_{y_j \in C_i} \left( y_j - \mu_i \right)^2`

where :math:`y_j` is the target value of data point j and :math:`\mu_i = \frac{1}{|C_i|}\sum_{y_j \in C_i} y_j` 
is mean of the target values within cluster :math:`C_i`. It measures the compactness (i.e goodness) of the clustering 
with respect to the target and should be as small as possible.


Model variance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We limit the model variance by discarding too complex models. We define the complexity of the clustering for each value of :math:`k` by its stability. 
The stability of each cluster :math:`i` in the clustering is measured by the average Jaccard Similarity between the original cluster :math:`A` and :math:`n` bootstrapped clusters :math:`B_b`:

:math:`JS_i(A|B) = \frac{\sum_{b=1}^n\frac{|A ∩ B_b|}{|A ∪ B_b|}}{n}`

A Jaccard similarity value <= 0.5 is indicative of a "dissolved cluster". Jaccard similarity values > 0.6 can be considered as an indicating patterns in the data and is suggested as the minimal threshold for stability.
However, which points exactly should belong to these clusters is not reliable. Generally, a valid, stable cluster should yield a mean Jaccard similarity value of 0.75 or more (*Hennig, 2008*). Only stable clusterings, i.e. clustering with low variance,
are considered as clustering candidates. Hence, the optimal number of clusters :math:`k` is the one yielding the minimum model bias, while having a stable clustering.
