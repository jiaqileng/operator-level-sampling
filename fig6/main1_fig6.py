# lindblad_1d_fft_rk4_overlap_unweighted.py
"""
Track the overlap
    ⟨√σ_β | ρ(t) | √σ_β⟩
where σ_β is the (discrete) Gibbs state associated with the potential V(x), using *unweighted*
discrete sums (i.e., no dx factors in normalization).

    Outputs:
    1) beta_is_<beta>.png
       - Line plot of overlap vs time with optional threshold annotations.
    2) beta_is_<beta>_data.npz
       - Saved (tgrid, overlap) arrays, uniformly downsampled to at most --save-n points.


    Author: Zherui Chen
    Date: 2025-10-21
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")     
import matplotlib.pyplot as plt
import time
from pathlib import Path

def fourier_diff_matrix(N: int, Ldom: float) -> np.ndarray:
    """
    Unitary-DFT-based first-derivative operator on a periodic domain of length Ldom.
    Returns a skew-Hermitian N N matrix D approximating d/dx.
    """
    n = np.arange(N)
    # Unitary DFT matrix
    F = np.exp(-1j * 2.0 * np.pi * np.outer(n, n) / N) / np.sqrt(N)

    if N % 2 == 0:
        pos = np.arange(0, N // 2)
        zero = np.array([0])
        neg = -np.arange(N // 2 - 1, 0, -1)
        k = np.concatenate([pos, zero, neg])
    else:
        half = (N - 1) // 2
        k = np.concatenate([np.arange(0, half + 1), -np.arange(half, 0, -1)])

    k = (2.0 * np.pi / Ldom) * k.astype(float)
    D = F.conj().T @ (1j * np.diag(k)) @ F
    return D  # skew-Hermitian

def main():
    tic = time.time()

    # ======== Parameters ========
    N    = 50                 # grid points (even N recommended)
    a, b = -2.0, 2.0          # domain [-2, 2]
    Ldom = b - a
    beta = 10.0                # inverse temperature β
    Tend = 10000.0              # final time
    dt   = 1e-4               # RK4 step

    # ======== Grid and operators ========
    dx = Ldom / N
    x  = np.linspace(a, b - dx, N)  # periodic grid (N points, last is b - dx)

    # Potential and its gradient
    V  = np.cos(np.pi * x) ** 2 + 0.25 * x ** 4
    dV = -np.pi * np.sin(2.0 * np.pi * x) + x ** 3  # dV/dx

    # Fourier spectral 1st-derivative matrix (periodic, unitary-DFT)
    D = fourier_diff_matrix(N, Ldom)  # skew-Hermitian, complex

    # Lindblad operator L and derived matrices (alpha = 0)
    W   = np.diag(dV)
    Lop = -1j * np.sqrt(2.0 / beta) * D - 1j * np.sqrt(beta / 2.0) * W
    K   = Lop.conj().T @ Lop

    # Lindbladian RHS
    def rhs(R: np.ndarray) -> np.ndarray:
        return Lop @ R @ Lop.conj().T - 0.5 * (K @ R + R @ K)

    # ======== Gibbs state σβ and |√σ> (NO dx in normalization) ========
    w          = np.exp(-beta * V)
    Z          = np.sum(w)          # NO dx
    sigma      = w / Z
    sqrt_sigma = np.sqrt(sigma)     # vector length N

    # ======== Initial state ρ(0) (NO dx in normalization) ========
    x0, s0 = -1.7, 0.01
    psi = np.exp(-((x - x0) ** 2) / (2.0 * s0 ** 2))
    psi = psi / np.linalg.norm(psi)            # sum |psi|^2 = 1
    rho = np.outer(psi, psi.conj())            # pure state density matrix (N×N)

    # ======== Time stepping (RK4) & overlap tracking ========
    Nt      = int(round(Tend / dt))
    tgrid   = np.linspace(0.0, Tend, Nt + 1)
    overlap = np.zeros(Nt + 1, dtype=float)

    # record overlap at t=0 (NO dx^2)
    overlap[0] = float(np.real(np.vdot(sqrt_sigma, rho @ sqrt_sigma)))

    # RK4 loop
    for n in range(Nt):
        k1 = rhs(rho)
        k2 = rhs(rho + 0.5 * dt * k1)
        k3 = rhs(rho + 0.5 * dt * k2)
        k4 = rhs(rho + dt * k3)
        rho = rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # enforce Hermiticity and unit trace (NO dx)
        rho = 0.5 * (rho + rho.conj().T)
        tr  = np.real(np.trace(rho))
        rho = rho / tr

        overlap[n + 1] = float(np.real(np.vdot(sqrt_sigma, rho @ sqrt_sigma)))

    # ======== Plot: time vs overlap with thresholds & annotations ========
    thr1 = 0.50
    thr2 = 0.95

    idx1 = np.argmax(overlap > thr1)
    if not (overlap > thr1).any():
        idx1 = None

    idx2 = np.argmax(overlap > thr2)
    if not (overlap > thr2).any():
        idx2 = None

    if idx1 is not None:
        t1, y1 = tgrid[idx1], overlap[idx1]
        print(f"First overlap > {thr1:.2f} at index {idx1}: t = {t1:.8f}")
    else:
        print(f"No overlap > {thr1:.2f} found in [0, Tend].")

    if idx2 is not None:
        t2, y2 = tgrid[idx2], overlap[idx2]
        print(f"First overlap > {thr2:.2f} at index {idx2}: t = {t2:.8f}")
    else:
        print(f"No overlap > {thr2:.2f} found in [0, Tend].")

    fig = plt.figure()
    plt.plot(tgrid, overlap, linewidth=1.8, label='overlap')
    plt.xlabel('time')
    plt.ylabel(r'$\langle \sqrt{\sigma_\beta}\,|\,\rho(t)\,|\,\sqrt{\sigma_\beta}\rangle$')
    plt.title(f'Overlap vs. time (unweighted sums) — β={beta:.3g}')
    plt.grid(True)

    # Annotations for thr1
    if idx1 is not None:
        plt.plot(t1, y1, 'o', markersize=8, label=f'>{thr1:.2f} at t={t1:.8f}')
        plt.axvline(t1, linestyle='--', color='k', alpha=0.5)
        plt.axhline(thr1, linestyle=':', color='k', alpha=0.5)
        plt.text(t1, y1, f'  t={t1:.8f}', va='bottom', fontsize=10)

    # Annotations for thr2
    if idx2 is not None:
        plt.plot(t2, y2, 's', markersize=8, label=f'>{thr2:.2f} at t={t2:.8f}')
        plt.axvline(t2, linestyle='--', color='k', alpha=0.5)
        plt.axhline(thr2, linestyle=':', color='k', alpha=0.5)
        plt.text(t2, y2, f'  t={t2:.8f}', va='top', fontsize=10)

    plt.legend()
    plt.tight_layout()

    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"beta_is_{beta:.6g}".replace(" ", "_")
    figfile = out_dir / f"{safe_name}.png"
    fig.savefig(figfile, dpi=300, bbox_inches="tight")  
    # plt.show()
    # np.savez(out_dir / f"{safe_name}_data.npz", tgrid=tgrid, overlap=overlap)
       
    save_n = 100000
    m = tgrid.size
    if m > save_n:
        idx = np.linspace(0, m - 1, save_n, dtype=int)  
    else:
        idx = np.arange(m)

    tgrid_save   = tgrid[idx]
    overlap_save = overlap[idx]

    np.savez(out_dir / f"{safe_name}_data.npz", tgrid=tgrid_save, overlap=overlap_save)
    print(f"Saved {tgrid_save.size} points (from original {m}).")


    print(f"Elapsed time: {time.time() - tic:.3f} s")

if __name__ == "__main__":
    main()