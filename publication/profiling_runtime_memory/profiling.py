import os
import gc
import time
import kmedoids
import argparse
import numpy as np
import pandas as pd

from numba import njit, prange
from memory_profiler import memory_usage

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

from fgclustering import ClusteringKMedoids, ClusteringClara, DistanceRandomForestProximity


def calculate_distance_matrix(model, X):
    distance_metric = DistanceRandomForestProximity()
    distance_metric.calculate_terminals(estimator=model, X=X)
    distance_metric.calculate_distance_matrix(sample_indices=np.arange(X.shape[0]))


def calculate_distance_matrix_memmap(model, X, dir_output):
    distance_metric = DistanceRandomForestProximity(memory_efficient=True, dir_distance_matrix=dir_output)
    distance_metric.calculate_terminals(estimator=model, X=X)
    distance_matrix, file = distance_metric.calculate_distance_matrix(sample_indices=np.arange(X.shape[0]))
    distance_metric.remove_distance_matrix(distance_matrix, file)


def run_clustering_kmedoids(model, X, k, method_clustering, seed):
    distance_metric = DistanceRandomForestProximity()
    distance_metric.calculate_terminals(estimator=model, X=X)

    clustering = ClusteringKMedoids(method=method_clustering, init="random", max_iter=100, random_state=seed)
    clustering.run_clustering(
        k=k,
        distance_metric=distance_metric,
        sample_indices=np.arange(X.shape[0]),
        random_state_subsampling=seed,
        verbose=0,
    )


def run_clustering_kmedoids_memmap(model, X, k, method_clustering, seed, dir_output):
    distance_metric = DistanceRandomForestProximity(memory_efficient=True, dir_distance_matrix=dir_output)
    distance_metric.calculate_terminals(estimator=model, X=X)

    clustering = ClusteringKMedoids(method=method_clustering, init="random", max_iter=100, random_state=seed)
    clustering.run_clustering(
        k=k,
        distance_metric=distance_metric,
        sample_indices=np.arange(X.shape[0]),
        random_state_subsampling=seed,
        verbose=0,
    )


def run_clustering_clara_memmap(model, X, k, method_clustering, seed, dir_output):
    distance_metric = DistanceRandomForestProximity(memory_efficient=True, dir_distance_matrix=dir_output)
    distance_metric.calculate_terminals(estimator=model, X=X)

    clustering = ClusteringClara(
        sub_sample_size=0.5,
        sampling_iter=5,
        method=method_clustering,
        init="random",
        max_iter=100,
        random_state=seed,
    )
    clustering.run_clustering(
        k=k,
        distance_metric=distance_metric,
        sample_indices=np.arange(X.shape[0]),
        random_state_subsampling=seed,
        verbose=0,
    )


def generate_dataset(n_samples, random_state):

    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=7,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        random_state=random_state,
    )

    print(f"Generated dataset with {len(y)} samples and {len(np.unique(y))} classes.")

    classifier = RandomForestClassifier(
        max_samples=0.8, bootstrap=True, oob_score=True, random_state=random_state
    )

    grid = {"max_depth": [5, 10], "max_features": ["sqrt"]}
    grid_classifier = GridSearchCV(classifier, grid, cv=5)
    grid_classifier.fit(X, y)
    model = grid_classifier.best_estimator_

    print(f"OOB score of the trained model: {model.oob_score_}")

    return X, y, model


def benchmark(
    X,
    y,
    model,
    k,
    number_samples,
    number_iterations,
    dir_output,
    seed,
    methods=None,
):
    if methods is None:
        methods = [
            "numpy_dist_mat",
            "numpy_pam",
            "numpy_fastpam1",
            "numpy_fasterpam",
            "memmap_dist_mat",
            "memmap_pam",
            "memmap_fastpam1",
            "memmap_fasterpam",
            "memmap_fasterpam_clara",
        ]

    runtime = {method: [] for method in methods}
    memory = {method: [] for method in methods}
    samples = []

    interval = 0.01

    method_dispatch = {
        "numpy_dist_mat": lambda X_sample: calculate_distance_matrix(model, X_sample),
        "numpy_pam": lambda X_sample: run_clustering_kmedoids(model, X_sample, k, "pam", seed),
        "numpy_fastpam1": lambda X_sample: run_clustering_kmedoids(model, X_sample, k, "fastpam1", seed),
        "numpy_fasterpam": lambda X_sample: run_clustering_kmedoids(model, X_sample, k, "fasterpam", seed),
        "memmap_dist_mat": lambda X_sample: calculate_distance_matrix_memmap(model, X_sample, dir_output),
        "memmap_pam": lambda X_sample: run_clustering_kmedoids_memmap(
            model, X_sample, k, "pam", seed, dir_output
        ),
        "memmap_fastpam1": lambda X_sample: run_clustering_kmedoids_memmap(
            model, X_sample, k, "fastpam1", seed, dir_output
        ),
        "memmap_fasterpam": lambda X_sample: run_clustering_kmedoids_memmap(
            model, X_sample, k, "fasterpam", seed, dir_output
        ),
        "memmap_fasterpam_clara": lambda X_sample: run_clustering_clara_memmap(
            model, X_sample, k, "fasterpam", seed, dir_output
        ),
    }

    for i in number_samples:
        for j in range(number_iterations):
            samples.append(i)
            X_sample = X[:i]

            for method in methods:
                print(f"[{method}] Benchmarking {i} samples (iteration {j + 1})")

                start_time = time.perf_counter()
                mem_usage = memory_usage(
                    (method_dispatch[method], (X_sample,)),
                    max_iterations=1,
                    interval=interval,
                )
                end_time = time.perf_counter()

                runtime[method].append(end_time - start_time)
                memory[method].append(np.mean(mem_usage) / 1024)  # convert to MB

                gc.collect()
                time.sleep(1)

    # Save results
    pd.DataFrame({**{"samples": samples}, **runtime}).to_csv(
        os.path.join(dir_output, "profiling_runtime.csv"), index=False
    )
    pd.DataFrame({**{"samples": samples}, **memory}).to_csv(
        os.path.join(dir_output, "profiling_memory.csv"), index=False
    )

    return (
        os.path.join(dir_output, "profiling_runtime.csv"),
        os.path.join(dir_output, "profiling_memory.csv"),
    )


def plot_benchmark_distance_matrix(file_runtime, file_memory, dir_output=None):
    # Load and aggregate data
    table_runtime = pd.read_csv(file_runtime).groupby("samples").agg(["mean", "std"])
    table_memory = pd.read_csv(file_memory).groupby("samples").agg(["mean", "std"])

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Common plotting parameters
    plot_params = dict(capsize=5, elinewidth=1, capthick=1, markeredgewidth=1, linestyle="--", linewidth=0.8)

    # --- Left plot: runtime distance matrix---
    ax1.errorbar(
        table_runtime.index,
        table_runtime["numpy_dist_mat"]["mean"],
        yerr=table_runtime["numpy_dist_mat"]["std"],
        label="Numpy",
        marker=">",
        color="seagreen",
        **plot_params,
    )
    ax1.errorbar(
        table_runtime.index,
        table_runtime["memmap_dist_mat"]["mean"],
        yerr=table_runtime["memmap_dist_mat"]["std"],
        label="Memmap",
        marker="<",
        color="mediumseagreen",
        **plot_params,
    )
    ax1.set_title("Runtime Comparison for Distance Matrix Computations")
    ax1.set_xlabel("Number of Samples (n)")
    ax1.set_ylabel("Runtime (seconds)")
    ax1.legend()

    # --- Right plot: memory distance matrix---
    ax2.errorbar(
        table_memory.index,
        table_memory["numpy_dist_mat"]["mean"],
        yerr=table_memory["numpy_dist_mat"]["std"],
        label="Numpy",
        marker=">",
        color="firebrick",
        **plot_params,
    )
    ax2.errorbar(
        table_memory.index,
        table_memory["memmap_dist_mat"]["mean"],
        yerr=table_memory["memmap_dist_mat"]["std"],
        label="Memmap",
        marker="<",
        color="red",
        **plot_params,
    )

    ax2.set_title("Memory Usage Comparison for Distance Matrix Computations")
    ax2.set_xlabel("Number of Samples (n)")
    ax2.set_ylabel("Memory (MB)")
    ax2.legend()

    plt.tight_layout()

    if dir_output:
        file_plot = os.path.join(dir_output, "profiling_distance_matrix.png")
        plt.savefig(file_plot, bbox_inches="tight", dpi=600)
    plt.show()


def plot_benchmark_cluster_methods(file_runtime, file_memory, dir_output=None):

    table_runtime = pd.read_csv(file_runtime).groupby("samples").agg(["mean", "std"])
    table_memory = pd.read_csv(file_memory).groupby("samples").agg(["mean", "std"])

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Common plotting parameters
    plot_params = dict(capsize=5, elinewidth=1, capthick=1, markeredgewidth=1, linestyle="--", linewidth=0.8)

    # --- Left plot: runtime clustering---
    ax1.errorbar(
        table_runtime.index,
        table_runtime["numpy_pam"]["mean"],
        yerr=table_runtime["numpy_pam"]["std"],
        label="Numpy PAM",
        marker="o",
        color="lightskyblue",
        **plot_params,
    )
    ax1.errorbar(
        table_runtime.index,
        table_runtime["numpy_fastpam1"]["mean"],
        yerr=table_runtime["numpy_fastpam1"]["std"],
        label="Numpy FastPAM1",
        marker="s",
        color="DeepSkyBlue",
        **plot_params,
    )
    ax1.errorbar(
        table_runtime.index,
        table_runtime["numpy_fasterpam"]["mean"],
        yerr=table_runtime["numpy_fasterpam"]["std"],
        label="Numpy FasterPAM",
        marker="^",
        color="RoyalBlue",
        **plot_params,
    )

    ax1.set_title("Runtime Comparison for Clustering Methods")
    ax1.set_xlabel("Number of Samples (n)")
    ax1.set_ylabel("Runtime (seconds)")
    ax1.legend()

    # --- Right plot: memory clustering---
    ax2.errorbar(
        table_memory.index,
        table_memory["numpy_pam"]["mean"],
        yerr=table_memory["numpy_pam"]["std"],
        label="Numpy PAM",
        marker="o",
        color="thistle",
        **plot_params,
    )
    ax2.errorbar(
        table_memory.index,
        table_memory["numpy_fastpam1"]["mean"],
        yerr=table_memory["numpy_fastpam1"]["std"],
        label="Numpy FastPAM1",
        marker="s",
        color="Orchid",
        **plot_params,
    )
    ax2.errorbar(
        table_memory.index,
        table_memory["numpy_fasterpam"]["mean"],
        yerr=table_memory["numpy_fasterpam"]["std"],
        label="Numpy FasterPAM",
        marker="^",
        color="DarkOrchid",
        **plot_params,
    )

    ax2.set_title("Memory Usage Comparison for Clustering Methods")
    ax2.set_xlabel("Number of Samples (n)")
    ax2.set_ylabel("Memory (MB)")
    ax2.legend()

    plt.tight_layout()

    if dir_output:
        file_plot = os.path.join(dir_output, "profiling_clustering_methods.png")
        plt.savefig(file_plot, bbox_inches="tight", dpi=600)
    plt.show()


def plot_benchmark_cluster_approaches(file_runtime, file_memory, dir_output=None):

    table_runtime = pd.read_csv(file_runtime).groupby("samples").agg(["mean", "std"])
    table_memory = pd.read_csv(file_memory).groupby("samples").agg(["mean", "std"])

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Common plotting parameters
    plot_params = dict(capsize=5, elinewidth=1, capthick=1, markeredgewidth=1, linestyle="--", linewidth=0.8)

    # --- Left plot: runtime clustering---
    ax1.errorbar(
        table_runtime.index,
        table_runtime["numpy_fasterpam"]["mean"],
        yerr=table_runtime["numpy_fasterpam"]["std"],
        label="Numpy FasterPAM",
        marker="^",
        color="SkyBlue",
        **plot_params,
    )
    ax1.errorbar(
        table_runtime.index,
        table_runtime["memmap_fasterpam"]["mean"],
        yerr=table_runtime["memmap_fasterpam"]["std"],
        label="Memmap FasterPAM",
        marker="D",
        color="DodgerBlue",
        **plot_params,
    )
    ax1.errorbar(
        table_runtime.index,
        table_runtime["memmap_fasterpam_clara"]["mean"],
        yerr=table_runtime["memmap_fasterpam_clara"]["std"],
        label="Memmap FasterPAM CLARA",
        marker="*",
        color="Navy",
        **plot_params,
    )

    ax1.set_title("Runtime Comparison for Clustering Approaches")
    ax1.set_xlabel("Number of Samples (n)")
    ax1.set_ylabel("Runtime (seconds)")
    ax1.legend()

    # --- Right plot: memory clustering---
    ax2.errorbar(
        table_memory.index,
        table_memory["numpy_fasterpam"]["mean"],
        yerr=table_memory["numpy_fasterpam"]["std"],
        label="Numpy FasterPAM",
        marker="^",
        color="Plum",
        **plot_params,
    )
    ax2.errorbar(
        table_memory.index,
        table_memory["memmap_fasterpam"]["mean"],
        yerr=table_memory["memmap_fasterpam"]["std"],
        label="Memmap FasterPAM",
        marker="D",
        color="MediumOrchid",
        **plot_params,
    )
    ax2.errorbar(
        table_memory.index,
        table_memory["memmap_fasterpam_clara"]["mean"],
        yerr=table_memory["memmap_fasterpam_clara"]["std"],
        label="Memmap FasterPAM CLARA",
        marker="*",
        color="indigo",
        **plot_params,
    )

    ax2.set_title("Memory Usage Comparison for Clustering Approaches")
    ax2.set_xlabel("Number of Samples (n)")
    ax2.set_ylabel("Memory (MB)")
    ax2.legend()

    plt.tight_layout()

    if dir_output:
        file_plot = os.path.join(dir_output, "profiling_clustering_approaches.png")
        plt.savefig(file_plot, bbox_inches="tight", dpi=600)
    plt.show()


def main():
    seed = 42

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Performance Benchmarks")
    parser.add_argument("--o", type=str, default="./", help="Output directory")
    parser.add_argument("--k", type=int, default=4, help="Number of clusters")
    parser.add_argument("--n", type=int, default=10, help="Number of clusters")
    args = parser.parse_args()

    k = args.k
    number_iterations = args.n
    dir_output = args.o
    os.makedirs(dir_output, exist_ok=True)

    print(f"Input parameters: --o={dir_output}, --k={k}")

    number_samples = [
        50,
        100,
        200,
        300,
        500,
        700,
        1000,
        2000,
        3000,
        5000,
        7000,
        10000,
        20000,
        30000,
        40000,
    ]

    X, y, model = generate_dataset(np.max(number_samples), seed)
    file_runtime, file_memory = benchmark(
        X,
        y,
        model,
        k,
        number_samples,
        number_iterations,
        dir_output,
        seed,
        methods=None,
    )
    plot_benchmark_distance_matrix(file_runtime, file_memory, dir_output)
    plot_benchmark_cluster_methods(file_runtime, file_memory, dir_output)
    plot_benchmark_cluster_approaches(file_runtime, file_memory, dir_output)


if __name__ == "__main__":
    main()
