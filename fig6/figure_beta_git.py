# pip install numpy scipy matplotlib h5py
"""
Plot overlap trajectories for multiple inverse temperatures β with a symlog time axis.


Inputs:
    "data_cluster/overlap_beta_2.mat",
    "data_cluster/overlap_beta_3.mat",
    "data_cluster/overlap_beta_4.mat",
    "data_cluster/overlap_beta_5.mat",
    "data_cluster/beta_is_10_data.npz"

Outputs: 
    Figure 6(b).



Author: Zherui Chen
Date: 2025-10-27

"""
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.unicode_minus": False,
    "font.size": 12,
})

def load_t_overlap_mat(path, t_key="tgrid", y_key="overlap"):
    
    try:
        d = loadmat(path, squeeze_me=True, struct_as_record=False)
        t = np.asarray(d[t_key]).ravel()
        y = np.asarray(d[y_key]).ravel()
    except Exception:
        import h5py
        with h5py.File(path, "r") as f:
            t = np.array(f[t_key]).squeeze()
            y = np.array(f[y_key]).squeeze()
    return t, y

def load_beta10_npz(path_npz):
    
    with np.load(path_npz, allow_pickle=True) as d:
        keys = d.files
        if "tgrid" in keys and "overlap" in keys:
            t = np.asarray(d["tgrid"]).ravel()
            y = np.asarray(d["overlap"]).ravel()
        elif "tgrid.npy" in keys and "overlap.npy" in keys:
            t = np.asarray(d["tgrid.npy"]).ravel()
            y = np.asarray(d["overlap.npy"]).ravel()
        else:
            one_d = [k for k in keys if d[k].ndim >= 1]
            if len(one_d) < 2:
                raise ValueError(f"Unexpected contents in {path_npz}: {keys}")
            t = np.asarray(d[one_d[0]]).ravel()
            y = np.asarray(d[one_d[1]]).ravel()
    return t, y

mat_files = [
    "overlap_beta_2.mat",
    "overlap_beta_3.mat",
    "overlap_beta_4.mat",
    "overlap_beta_5.mat",
]

fig, ax = plt.subplots(figsize=(7.6, 4.4))

t_upper = 1600
for fn in mat_files:
    t, y = load_t_overlap_mat(fn)
    m = (t >= 0) & (t <= t_upper)
    t, y = np.asarray(t[m]), np.asarray(y[m])
    order = np.argsort(t)
    t, y = t[order], y[order]

    m_beta = re.search(r"beta[_-]?(\d+)", fn)
    label = rf"$\beta={m_beta.group(1)}$" if m_beta else fn
    ax.plot(t, np.sqrt(y), lw=1.6, label=label)

t10, y10 = load_beta10_npz("data_cluster/beta_is_10_data.npz")
m10 = (t10 >= 0) & (t10 <= t_upper)
t10, y10 = np.asarray(t10[m10]), np.asarray(y10[m10])
ord10 = np.argsort(t10)
t10, y10 = t10[ord10], y10[ord10]
ax.plot(t10, np.sqrt(y10), lw=1.8, label=rf"$\beta=10$")

threshold_x = 1 
ax.set_xscale("symlog", linthresh=threshold_x, linscale=0.15, base=10) 
ax.set_xlim(0, t_upper)
ax.set_ylim(0, 1)

ax.set_xlabel(r"$t$")
ax.set_ylabel("overlap")


xticks_linear = [0, 1, 2, 3]
xticks_log = [5, 10, 20, 50, 100, 200, 400, 1000]  
ax.set_xticks(xticks_linear + xticks_log)
ax.set_xticklabels([str(x) for x in (xticks_linear + xticks_log)])


ax.axvspan(threshold_x, t_upper, facecolor="0.92", zorder=0)


ax.axvline(threshold_x, ls="--", lw=1.2, color="k")


ax.annotate("log-scale region",
            xy=(threshold_x, 0.02), xycoords=("data", "axes fraction"),
            xytext=(8, 0), textcoords="offset points",
            ha="left", va="bottom", fontsize=14)

ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="best")
plt.tight_layout()
plt.show()

