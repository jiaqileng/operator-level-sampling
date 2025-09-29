#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Visualization of inverse spectral gaps vs. β.
draw Fig. 5 (b) of [Operator-Level Quantum Acceleration of Non-Logconcave Sampling]

This script reads four .mat files produced by MATLAB and plots 1/Gap:
    spectral_gap.mat
    spectral_gap_cla.mat
    spectral_gap_witten_lap.mat
    spectral_gap_Lx_witten.mat
All are expected under a 'data/' folder next to THIS script. 

Outputs:
    log-beta-gap.pdf 


    Author: Zherui Chen
    Date: 2025-02-24
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat

# ---- LaTeX rendering (set to False if TeX is not installed) ----
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}\usepackage{mathrsfs}'

# ---- Resolve data directory relative to this script ----
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def _ensure_exists(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Expected file not found: {p}\n"
                                f"Script dir: {BASE_DIR}\n"
                                f"Data dir:   {DATA_DIR}\n"
                                f"Tip: Run this script from anywhere; it always reads from {DATA_DIR}")

def load_gap(filename, varname, beta_fallback=None):
    """Load a gap vector and beta from a .mat file located in DATA_DIR.

    If the .mat is v7.3 (HDF5) and SciPy cannot load it, falls back to h5py.
    """
    path = DATA_DIR / filename
    _ensure_exists(path)

    try:
        m = loadmat(path)
        gap = np.asarray(m[varname]).squeeze()
        beta = np.asarray(m.get('beta', beta_fallback)).squeeze()
        return beta, gap
    except NotImplementedError:
        # Likely MATLAB v7.3 .mat; fall back to h5py
        import h5py
        with h5py.File(path, "r") as f:
            def read(name):
                # h5py stores datasets as column-major; transpose if needed
                arr = np.array(f[name])
                return np.squeeze(arr)
            gap = read(varname)
            beta = read("beta") if "beta" in f else beta_fallback
            beta = np.squeeze(np.array(beta))
            gap = np.squeeze(np.array(gap))
            return beta, gap

# ---- Load data ----
beta, spectral_gap            = load_gap("spectral_gap.mat",            "spectral_gap")
_,    spectral_gap_cla        = load_gap("spectral_gap_cla.mat",        "spectral_gap_cla",        beta_fallback=beta)
_,    spectral_gap_witten_lap = load_gap("spectral_gap_witten_lap.mat", "spectral_gap_witten_lap", beta_fallback=beta)
_,    spectral_gap_Lx_witten  = load_gap("spectral_gap_Lx_witten.mat",  "spectral_gap_Lx_witten",  beta_fallback=beta)

# ---- Avoid division-by-zero ----
eps = 1e-15
g1 = np.maximum(spectral_gap_witten_lap, eps)
g2 = np.maximum(spectral_gap_Lx_witten,  eps)
g3 = np.maximum(spectral_gap_cla,        eps)
g4 = np.maximum(spectral_gap,            eps)

# ---- Plot ----
plt.figure(figsize=(5, 3))
plt.semilogy(beta, 1.0 / g1, '-',  linewidth=2, color=[0.1, 0.6, 0.1], label=r'LD')
plt.semilogy(beta, 1.0 / g2, '--', linewidth=2, color=[0.2, 0.4, 0.8], label=r'Quantum-accelerated~LD')
plt.semilogy(beta, 1.0 / g3, '-.', linewidth=2, color=[0.8, 0.4, 0.2], label=r'RELD')
plt.semilogy(beta, 1.0 / g4, ':',  linewidth=2, color=[0.7, 0.2, 0.7], label=r'Quantum-accelerated~RE')

plt.xlabel(r'$\beta$', fontname='Times New Roman', fontsize=14)
plt.ylabel(r'$1/\mathrm{Gap}$', fontname='Times New Roman', fontsize=14)
plt.ylim([0, 100])
plt.title(r'$\beta^\prime = 1,\ \ \mu = 1$', fontsize=14)
plt.legend(loc='upper left', prop={'family': 'Times New Roman', 'size': 12})
plt.tight_layout()
plt.savefig(BASE_DIR / 'log-beta-gap.pdf', format='pdf')
plt.show()

