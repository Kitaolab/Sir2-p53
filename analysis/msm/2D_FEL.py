import pickle
import numpy as np
import pyemma
from matplotlib import pyplot as plt

# === Constants ===
T = 298.15  # Kelvin
_RT = -8.314 * T / 1000 / 4.184  # kcal/mol
lag_time = 40
dt_ps = "1 ps"
nclusters = 1000
cluster_pkl_path = "out_ll/cluster_trial_3d.pkl"

# === Load cluster and MSM ===
with open(cluster_pkl_path, "rb") as f:
    cluster_dict = pickle.load(f)

cluster = cluster_dict[f"n_clusters{nclusters}"]
msm = pyemma.msm.estimate_markov_model(cluster.dtrajs, lag=lag_time, dt_traj=dt_ps)

# === Load precomputed area1 and area2 ===
area1 = np.load("area1_all.npy")
area2 = np.load("area2_all.npy")

# === Plot Free Energy Landscape ===
fig, ax = plt.subplots(figsize=(10, 5))

pyemma.plots.plot_free_energy(
    area1,
    area2,
    weights=np.concatenate(msm.trajectory_weights()),
    kT=_RT * (-1 * 4.184),
    cbar_label="Free energy [kcal/mol]",
    ax=ax,
)

ax.scatter(cluster.clustercenters[:, 0], cluster.clustercenters[:, 1],
           color="black", s=5, alpha=0)

ax.set_xlabel("S1' ($nm^2$)", fontsize=14)
ax.set_ylabel("S2' ($nm^2$)", fontsize=14)
ax.tick_params(labelsize=12)
ax.legend().set_visible(False)

plt.tight_layout()
plt.savefig("2D_FEL.png", dpi=300)
plt.show()

