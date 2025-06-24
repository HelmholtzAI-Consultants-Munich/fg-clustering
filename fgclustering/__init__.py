"""
Forest-Guided Clustering (FGC) is an explainability method for Random Forest models that addresses one of the key limitations of many standard XAI techniques: the inability to effectively handle correlated features and complex decision patterns.
"""

from .clustering import ClusteringKMedoids, ClusteringClara

from .distance import DistanceRandomForestProximity, DistanceJensenShannon, DistanceWasserstein

from .forest_guided_clustering import (
    forest_guided_clustering,
    forest_guided_feature_importance,
    plot_forest_guided_feature_importance,
    plot_forest_guided_decision_paths,
)

from .optimizer import Optimizer

from .statistics import FeatureImportance


__all__ = [
    "ClusteringKMedoids",
    "ClusteringClara",
    "DistanceRandomForestProximity",
    "DistanceJensenShannon",
    "DistanceWasserstein",
    "forest_guided_clustering",
    "forest_guided_feature_importance",
    "plot_forest_guided_feature_importance",
    "plot_forest_guided_decision_paths",
    "Optimizer",
    "FeatureImportance",
]
