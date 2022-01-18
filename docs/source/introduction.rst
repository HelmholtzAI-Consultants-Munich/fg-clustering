Introduction
===============

Forest guided clustering



For correlated features common interpretation methods fail
Feature importance based methods e.g. pinpoint individual contribution of features.
But what if features depend on each other?

For example men have higher risk of a disease when young, but women have higher risk when old.
This kind of correlated pattern cannot be uncovered by feature importance metrics!

Forest-guided clustering allows to interpret correlated features
In order to be able to interpret a random forest model in the presence of strong correlations, forest-guided clustering groups data into subgroups that follow similar decision rules/feature pattern.

The algorithm thereby consistents of the following steps:
computation of a proximity matrix based on the random forest that defines what data points the random forest sees a similar.
based on this matrix the datapoints are sortet into clusters of data points that the random forest treasts similarly
these clusters are then visualized.

Normally for example for a binary classification the random forest separates the data points into exactly two groups, according to their prediction.
However there might be subgroups so that some of the dataponts which have prediction 1 are soreted into that category for completely different reasons
than other datapoints that are sortet into the same category.
fg clustering finds these subgroups which makes interpretation more feasible. and allows better to differentiate why certain data points are predicted for what they are.

(feature importance can then be inspected for the different subgroups separately.)

