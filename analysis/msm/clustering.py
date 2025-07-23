import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pyemma
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For headless environments

# Parameters

@dataclass(frozen=True)
class Parameters:
    area1_path: str
    area2_path: str
    output_directory: str
    dt: int
    n_clusters_for_try: List[int]
    lags_for_try: List[int]
    show_picture: bool = False


params = Parameters(
    area1_path="area1_all.npy",
    area2_path="area2_all.npy",
    output_directory="./out_ll",
    dt=1,
    n_clusters_for_try=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    lags_for_try=list(range(1, 51, 2)),
    show_picture=False,
)

Path(params.output_directory).mkdir(parents=True, exist_ok=True)


# ==============================
# Function: Plot ITS (3D)
# ==============================

def plot_its_3d(params: Parameters, max_iter: int = 200, nits: int = 4, n_jobs: int = 1) -> None:
    dict_cluster_trial_3d = {}

    print("Loading features...")
    area1 = np.load(params.area1_path)
    area2 = np.load(params.area2_path)
    features_3d = np.column_stack((area1, area2))

    for n_clusters in params.n_clusters_for_try:
        print(f"Clustering with {n_clusters} clusters...")
        clusters = pyemma.coordinates.cluster_kmeans(features_3d, k=n_clusters, max_iter=max_iter)
        dict_cluster_trial_3d[f"n_clusters_{n_clusters}"] = clusters

        print("Computing implied timescales...")
        its = pyemma.msm.its(clusters.dtrajs, lags=params.lags_for_try, nits=nits, errors='bayes', n_jobs=n_jobs)

        # Plot ITS
        fig, ax = plt.subplots()
        pyemma.plots.plot_implied_timescales(its, units='ps', dt=params.dt, ylog=True)
        fig.tight_layout()
        plt.savefig(f"{params.output_directory}/its_cluster{n_clusters}_3d.png", dpi=300)

        if params.show_picture:
            plt.show()

        plt.clf()
        plt.close()

    # Save clustering result
    with open(f"{params.output_directory}/cluster_trial_3d.pkl", 'wb') as f:
        pickle.dump(dict_cluster_trial_3d, f)


# Main Execution

if __name__ == "__main__":
    plot_its_3d(params=params, max_iter=2000, nits=4, n_jobs=1)

