"""
Forest-Guided Clustering (FGC) is an explainability method for Random Forest models that addresses one of the key limitations of many standard XAI techniques: the inability to effectively handle correlated features and complex decision patterns.
"""

from .forest_guided_clustering import (
    forest_guided_clustering,
    forest_guided_feature_importance,
    plot_forest_guided_feature_importance,
    plot_forest_guided_decision_paths,
)

from .clustering import ClusteringKMedoids, ClusteringClara

from .distance import DistanceRandomForestProximity, DistanceJensenShannon, DistanceWasserstein

from .optimizer import Optimizer

from .statistics import FeatureImportance


__all__ = [
    "forest_guided_clustering",
    "forest_guided_feature_importance",
    "plot_forest_guided_feature_importance",
    "plot_forest_guided_decision_paths",
    "ClusteringKMedoids",
    "ClusteringClara",
    "DistanceRandomForestProximity",
    "DistanceJensenShannon",
    "DistanceWasserstein",
    "Optimizer",
    "FeatureImportance",
]
