# Operator-Level Quantum Acceleration of Non-Logconcave Sampling


This repository provides two self-contained folders—**fig4/** and **fig5/**—for reproducing the figures from [*Operator-Level Quantum Acceleration of Non-Logconcave Sampling*](https://arxiv.org/abs/2505.05301) ([PDF](https://arxiv.org/pdf/2505.05301)).

---

## `fig4/` — MALA vs. Quantum-accelerated LD (with Gibbs) on Müller–Brown potential and panel visualization

**Contents**

* `QSVT_filter_construction/`
  Helper routines used by the quantum-accelerated pipeline:

  * `smoothStep.m`
  * `chebyshev_coeffs_fft.m`
  * `filter_total.m`
* `Q_accelerated_LD.m`
  Runs the quantum-accelerated Langevin dynamics (QSVT-based pipeline).
  Saves to `data/`:

  * `beta04QSVT.mat`, `beta06QSVT.mat`, `beta08QSVT.mat` (variable: `probability_final`)
  * `gibbs_04.mat`, `gibbs_06.mat`, `gibbs_08.mat` (variable: true distribution `gibbs_state`)
  * `All_Overlap_QSVT.mat` (variable: `All_Overlap`, 1×3)
* `MALA.m`
  Runs MALA sampling for β ∈ {0.4, 0.6, 0.8}.
  Saves to `data/`:

  * `beta04MALA.mat`, `beta06MALA.mat`, `beta08MALA.mat` (variable: `density_storage`)
  * `All_Overlap_MALA.mat` (variable: `All_Overlap`, 1×3)
* `draw_fig4.m`
  Loads the above `.mat` files and draws the 3×3 panel (MALA / QSVT / Gibbs).

* `data/`
  Created automatically after you run the MATLAB scripts.

**How to run (order)**

1. Open MATLAB and `cd` into `fig4/`.
2. Run `Q_accelerated_LD.m` (this creates the QSVT and Gibbs files in `data/`).
3. Run `MALA.m` (this creates the MALA files and `All_Overlap_MALA.mat`).
4. Run `draw_fig4.m` to generate the figure.

   * Tip: If `parfor` is unavailable, change it to `for` in the scripts.

---

## `fig5/` — Inverse spectral gaps vs. β for different dynamics

**Contents**

* `main_fig_5.m` (MATLAB)
  Computes spectral gaps for a range of β and **stores results only** (no plotting).
  Saves to `data/` (each file also contains the vector `beta`):

  * `spectral_gap.mat`              (variable: `spectral_gap`)
  * `spectral_gap_cla.mat`          (variable: `spectral_gap_cla`)
  * `spectral_gap_witten_lap.mat`   (variable: `spectral_gap_witten_lap`)
  * `spectral_gap_Lx_witten.mat`    (variable: `spectral_gap_Lx_witten`)
* `draw_fig5.py` (Python)
  Reads the `.mat` files under `data/` and plots `1/Gap` on a semilog-y axis, writing `log-beta-gap.pdf`.

  * Requires: `numpy`, `matplotlib`, `scipy` (and optionally LaTeX if `usetex=True`).
* `log-beta-gap.pdf`
  Produced by `draw_fig5.py`.
* `data/`
  Output folder created by `main_fig_5.m`.

**How to run (order)**

1. Open MATLAB and `cd` into `fig5/`, then run:

   ```matlab
   main_fig_5
   ```

   This will populate `fig5/data/` with the four `.mat` files.
2. Run the Python plotter:

   ```bash
   # From the same folder (or anywhere):
   python draw_fig5.py
   ```

   The script saves `log-beta-gap.pdf` next to itself.

---


## `fig6/` Lindbladian-based warm-start generation —  (generate data + plot)

This folder contains scripts to (i) run the Lindblad dynamics for chosen inverse temperatures **β** and save the time–overlap data, and (ii) generate the final figure from the saved outputs.

---

**Contents**

- `main1_fig6.py` (Python)  
  Runs the simulation for a specified **β**, saves overlap data and (optionally) a quick diagnostic plot.  
  You typically run this script **multiple times**, changing β each time.

- `figure_beta_git.py` (Python)  
  Reads the saved results produced by `main1_fig6.py` (for multiple β values) and generates the final figure for the paper / GitHub.


- `*.pdf` (outputs)  
  Final figure produced by `figure_beta_git.py` (filename depends on the script settings).

---

### Requirements

Install dependencies (recommended: use a virtual environment):

```bash
pip install numpy scipy matplotlib h5py
